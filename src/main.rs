//! U-D分解フィルタを非線形カルマンフィルタに適用する．
//! 
//! 問題設定は「カルマンフィルタの基礎」pp.174-182の例題と同じ．

// ud-filter-optからの変更点は，predictの先頭で入出力行列を線形化している点と，
// filtering内で観測値と出力推定値の差分をy-H*xからy-h(x)に書き換えた点のみ．

use std::fs;
use std::io::{Write, BufWriter};
use rand::distributions::{Distribution, Normal};
use std::mem::MaybeUninit;

// --------- 入出力数 --------- //
// 以下の状態空間モデルを考える．
// x[k+1] = f(x[k]) + G*w[k],
// y[k]   = h(x[k]) + v[k].
// このとき，行列F, G, Hのサイズはそれぞれ
// F: N×N, G: N×R, H: P×N
// となる．

/// 状態変数の個数
const SYS_N: usize = 3;

/// 入力数
const SYS_M: usize = 1;

/// 出力数
const SYS_P: usize = 1;
// ---------------------------- //

// -- ベクトル・行列型の定義 -- //
// Vector○: X次元ベクトル
type VectorN<T>  = [T; SYS_N];
type VectorP<T>  = [T; SYS_P];
// Matrix○x□: ○行□列行列
type MatrixNxN<T>  = [[T; SYS_N]; SYS_N];
type MatrixNxM<T>  = [[T; SYS_M]; SYS_N];
type MatrixPxN<T>  = [[T; SYS_N]; SYS_P];
type MatrixMxM<T>  = [[T; SYS_M]; SYS_M];
type MatrixNxNM<T> = [[T; SYS_N + SYS_M]; SYS_N];  // N×(N+M)
// ---------------------------- //

const DT: f64 = 0.5;  // オイラー法離散化周期 [s]
const END_TIME: f64 = 30.0; // [s]
const N: usize = (END_TIME / DT) as usize + 1;

// 問題設定
const RHO: f64 = 1.23;
const GRAVITY: f64 = 9.81;
const ETA: f64 = 6e3;
const M: f64 = 3e4;  // レーダと物体の間の水平距離
const ALTITUDE: f64 = 3e4;  // レーダの高度


fn main() {
    // CSVファイルにデータ保存
    // 同一ファイルが存在したら上書き
    let mut file = BufWriter::new( fs::File::create("result.csv").unwrap() );

    // 標準正規分布の乱数を生成
    let randn = Normal::new(0.0, 1.0);  // 平均値:0，標準偏差:1

    // --- 各行列を定義 --- //
    let p = [
        [9e3, 0.0, 0.0],
        [0.0, 4e5, 0.0],
        [0.0, 0.0, 0.4]
    ];
    let g = [[0.0]; 3];
    let q = [[0.0]];
    let r = [4e3];
    let mut ekf = ExUdFilter::new(p, g, q, r);

    let mut x = [90000.0, -6000.0, 0.003];
    ekf.x = x.clone();
    let y_true = calc_h(x);
    let y = y_true[0] + randn.sample(&mut rand::thread_rng()) * r[0].sqrt();

    // --------- データ書き込み ---------- //
    // 時刻
    file.write( "0.000,".as_bytes() ).unwrap();
    // 出力の真値・観測値
    file.write( format!("{:.7},{:.7},", y_true[0], y).as_bytes() ).unwrap();
    // 状態変数の真値
    file.write( format!("{:.7},{:.7},{:.7},", x[0], x[1], x[2]).as_bytes() ).unwrap();
    // EKF推定値
    file.write( format!("{:.7},{:.7},{:.7}\n", ekf.x[0], ekf.x[1], ekf.x[2]).as_bytes() ).unwrap();
    // ----------------------------------- //

    for t in 1..N {
        x = calc_f(x);
        let y_true = calc_h(x);
        let y = y_true[0] + randn.sample(&mut rand::thread_rng()) * r[0].sqrt();

        ekf.predict();
        ekf.filtering(&[y]);

        // --------- データ書き込み ---------- //
        // 時刻
        file.write( format!("{:.3},", DT * t as f64).as_bytes() ).unwrap();
        // 出力の真値・観測値
        file.write( format!("{:.7},{:.7},", y_true[0], y).as_bytes() ).unwrap();
        // 状態変数の真値
        file.write( format!("{:.7},{:.7},{:.7},", x[0], x[1], x[2]).as_bytes() ).unwrap();
        // EKF推定値
        file.write( format!("{:.7},{:.7},{:.7}\n", ekf.x[0], ekf.x[1], ekf.x[2]).as_bytes() ).unwrap();
        // ----------------------------------- //
    }
}

/// 拡張U-D分解フィルタ（拡張カルマンフィルタと同じ）
#[allow(non_snake_case)]
struct ExUdFilter {
    pub x: VectorN<f64>,    // 状態変数
    pub U: MatrixNxN<f64>,  // U-D分解した共分散行列
    F: MatrixNxN<f64>,  // システム行列
    G: MatrixNxM<f64>,  // 入力行列
    H: MatrixPxN<f64>,  // 出力行列
    R: VectorP<f64>,    // 観測ノイズの共分散行列の対角成分
}

impl ExUdFilter {
    /// 誤差共分散行列の初期値を零行列にするとU-D分解に失敗するので，
    /// スカラー行列にするのが無難．
    /// 
    /// 状態変数は全て零で初期化する．状態変数xはパブリックメンバにしているので，
    /// 零以外にしたい場合は構造体を作った後に適宜アクセスして書き換えること．
    #[allow(non_snake_case)]
    pub fn new(
        P: MatrixNxN<f64>,  // 共分散行列の初期値
        G: MatrixNxM<f64>,  // 入力行列
        Q: MatrixMxM<f64>,  // システムノイズの共分散行列
        R: VectorP<f64>     // 観測ノイズの共分散行列の対角成分
    ) -> Self {
        // システムノイズの分散Qが単位行列で無い場合には，
        // QをQ=C*C^Tと分解し，Gを改めてG*Cとおく．
        let c = cholesky_decomp(Q);
        let mut gc: MatrixNxM<f64> = unsafe {MaybeUninit::uninit().assume_init()};
        for i in 0..SYS_N {
            for j in 0..SYS_M {
                let mut sum = 0.0;
                for k in 0..SYS_M {
                    sum += G[i][k] * c[k][j];
                }
                gc[i][j] = sum;
            }
        }
        Self {
            x: [0.0; SYS_N],
            U: ud_decomp(P),
            F: [[0.0; SYS_N]; SYS_N],
            G: gc,
            H: [[0.0; SYS_N]; SYS_P],
            R: R,
        }
    }

    /// 予測ステップ
    /// 
    /// x(k+1) = F * x(k)
    /// P_bar = F*P*F^T + G*Q*Q^T
    pub fn predict(&mut self) {
        // Working array
        let mut qq: VectorN<f64>    = unsafe {MaybeUninit::uninit().assume_init()};
        let mut z:  VectorN<f64>    = unsafe {MaybeUninit::uninit().assume_init()};
        let mut w:  MatrixNxNM<f64> = unsafe {MaybeUninit::uninit().assume_init()};

        // 線形化と状態変数の更新
        self.calc_jacobian_f();
        self.x = calc_f(self.x);
        self.calc_jacobian_h();

        // qqとwの左NxN要素を初期化
        for j in (1..SYS_N).rev() {
            qq[j] = self.U[j][j];
            for i in 0..SYS_N {
                let mut sum = self.F[i][j];
                for k in 0..j {
                    sum += self.F[i][k] * self.U[k][j];
                }
                w[i][j] = sum;
            }
        }
        qq[0] = self.U[0][0];
        // wの右NxR要素を初期化
        for i in 0..SYS_N {
            for j in 0..SYS_M {
                w[i][j + SYS_N] = self.G[i][j];
            }
            w[i][0] = self.F[i][0];
        }
        // --- ここまででw, qq, self.xを計算

        for j in (1..SYS_N).rev() {
            let mut sum = 0.0;
            for k in 0..SYS_N {
                z[k] = w[j][k] * qq[k];
                sum += z[k] * w[j][k];
            }
            for k in SYS_N..(SYS_N + SYS_M) {
                sum += w[j][k] * w[j][k];
            }
            self.U[j][j] = sum;
            let u_recip = self.U[j][j].recip();
            for i in 0..j {
                sum = 0.0;
                for k in 0..SYS_N {
                    sum += w[i][k] * z[k];
                }
                for k in SYS_N..(SYS_N + SYS_M) {
                    sum += w[i][k] * w[j][k];
                }

                sum *= u_recip;
                for k in 0..(SYS_N + SYS_M) {
                    w[i][k] -= sum * w[j][k];
                }
                self.U[i][j] = sum;
            }
        }
        let mut sum = 0.0;
        for k in 0..SYS_N {
            sum += qq[k] * (w[0][k] * w[0][k]);  // qqには更新前のUの対角要素が入っている
        }
        for k in SYS_N..(SYS_N + SYS_M) {
            sum += w[0][k] * w[0][k];
        }
        self.U[0][0] = sum;
    }

    /// フィルタリングステップ
    pub fn filtering(&mut self, y: &VectorP<f64>) {
        // Working array
        let mut ff: VectorN<f64> = unsafe {MaybeUninit::uninit().assume_init()};  // U^T H^T
        let mut gg: VectorN<f64> = unsafe {MaybeUninit::uninit().assume_init()};  // D U^T H^T

        let yhat = calc_h(self.x);
        // 出力の数だけループ
        for l in 0..SYS_P {
            // y_diff := y - h(x)
            let mut y_diff = y[l] - yhat[l];
            
            for j in (1..SYS_N).rev() {
                ff[j] = self.H[l][j];
                for k in 0..j {
                    ff[j] += self.U[k][j] * self.H[l][k];
                }
                gg[j] = self.U[j][j] * ff[j];
            }
            ff[0] = self.H[l][0];
            gg[0] = self.U[0][0] * ff[0];
            // --- ここまででy_diff, ff, ggを計算

            let mut alpha = self.R[l] + ff[0] * gg[0];  // 式 8.46
            let mut gamma = alpha.recip();
            self.U[0][0] = self.R[l] * gamma * self.U[0][0];  // 式 8.46
            for j in 1..SYS_N {
                let mut beta = alpha;
                alpha += ff[j] * gg[j];  // 式 8.47
                let lambda = ff[j] * gamma;  // 式　8.49
                gamma = alpha.recip();
                self.U[j][j] = beta * self.U[j][j] * gamma;  // 式 8.48
                for i in 0..j {
                    beta = self.U[i][j];
                    self.U[i][j] -= lambda * gg[i];  // 式 8.50
                    gg[i] +=  beta * gg[j];  // 式 8.51
                }
            }
            y_diff *= gamma;
            for j in 0..SYS_N {
                self.x[j] += gg[j] * y_diff;
            }
        }
    }

    fn calc_jacobian_f(&mut self) {
        let tmp = DT * RHO * (-self.x[0] / ETA).exp() * self.x[1];
        self.F[0][0] = 1.0;
        self.F[0][1] = DT;
        self.F[0][2] = 0.0;
        self.F[1][0] = -(0.5 / ETA) * tmp * self.x[1] * self.x[2];
        self.F[1][1] = 1.0 + tmp * self.x[2];
        self.F[1][2] = 0.5 * tmp * self.x[1];
        self.F[2][0] = 0.0;
        self.F[2][1] = 0.0;
        self.F[2][2] = 1.0;
    }
    
    fn calc_jacobian_h(&mut self) {
        let tmp = self.x[0] - ALTITUDE;
        self.H[0][0] = tmp / (M*M + tmp*tmp).sqrt();
        self.H[0][1] = 0.0;
        self.H[0][2] = 0.0;
    }
}


/// U-D分解（P = U * D * U^T）
/// 
/// * Pをn×n非負正定値対称行列とする．
/// * Uは対角成分を1とするn×n上三角行列．
/// * Dはn×n対角行列．
/// 
/// 返り値は，対角成分をDとし，それ以外の要素をUとした上三角行列．
fn ud_decomp(mut p: MatrixNxN<f64>) -> MatrixNxN<f64> {
    let mut ud: MatrixNxN<f64> = unsafe {MaybeUninit::uninit().assume_init()};

    for k in (1..SYS_N).rev() {  // n-1, n-2, ..., 1
        ud[k][k] = p[k][k];
        let ud_recip = ud[k][k].recip();
        for j in 0..k {
            ud[j][k] = p[j][k] * ud_recip;
            ud[k][j] = 0.0;  // 対角を除いた下三角成分を0埋め

            let tmp = ud[j][k] * ud[k][k];
            for i in 0..=j {  // 両側閉区間
                p[i][j] -= ud[i][k] * tmp;
            }
        }
    }
    ud[0][0] = p[0][0];  // pを書き換えてるから，d[0]の代入はこの位置じゃないとダメ

    ud
}

/// コレスキー分解（P = U * U^T）
/// 
/// * Pをn×n非負正定値対称行列とする．
/// * Uは対角要素が非負の値をとるn×n上三角行列．
fn cholesky_decomp(mut p: MatrixMxM<f64>) -> MatrixMxM<f64> {
    let mut u: MatrixMxM<f64> = unsafe {MaybeUninit::uninit().assume_init()};

    for k in (1..SYS_M).rev() {
        u[k][k] = p[k][k].sqrt();
        let u_recip = u[k][k].recip();
        for j in 0..k {
            u[j][k] = p[j][k] * u_recip;
            u[k][j] = 0.0;  // 対角を除いた下三角成分を0埋め
            for i in 0..=j {
                p[i][j] -= u[i][k] * u[j][k];
            }
        }
    }
    u[0][0] = p[0][0].sqrt();

    u
}

#[allow(dead_code)]
/// MatrixNxNを整形してプロット
fn plot_nn(m: &MatrixNxN<f64>) {
    println!("MatrixNxN = [");
    for i in 0..SYS_N {
        print!("    [");
        for j in 0..SYS_N {
            print!("{:.12}", m[i][j]);
            if j < (SYS_N - 1) {
                print!(", ");
            }
        }
        println!("],");
    }
    println!("];");
}

fn calc_f(x: VectorN<f64>) -> VectorN<f64> {
    [
        x[0] + DT * x[1],
        x[1] + DT * ( 0.5*RHO* (-x[0] / ETA).exp() * x[1] * x[1] * x[2] - GRAVITY ),
        x[2]
    ]
}

fn calc_h(x: VectorN<f64>) -> VectorP<f64> {
    let tmp = x[0] - ALTITUDE;
    [(M * M + tmp * tmp).sqrt()]
}