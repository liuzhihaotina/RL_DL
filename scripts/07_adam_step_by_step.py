"""手把手：loss = w²，一个参数，逐步打印 SGD 和 Adam 的每一个中间量。

只跑 3 步，所有数字都能手算核对。学习率 0.1，起点 w = 1.0。
梯度：g = dL/dw = 2w。
"""

from __future__ import annotations

import argparse
import math

LR = 0.1
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    w_sgd = 1.0
    w_adam = 1.0
    m = 0.0  # 一阶矩：梯度的滑动平均
    v = 0.0  # 二阶矩：梯度平方的滑动平均

    print("起点：w = 1.0，loss = 1.0，梯度 g = 2w = 2.0\n")

    for t in range(1, args.steps + 1):
        # ---------- SGD ----------
        g_sgd = 2 * w_sgd
        sgd_step = LR * g_sgd
        print(f"===== 第 {t} 步 =====")
        print(f"[SGD ] 当前 w = {w_sgd:.4f}，梯度 g = 2w = {g_sgd:.4f}")
        print(f"[SGD ] 步长 = lr × g = 0.1 × {g_sgd:.4f} = {sgd_step:.4f}")
        w_sgd -= sgd_step
        print(f"[SGD ] 新 w = {w_sgd:.4f}，loss = {w_sgd**2:.6f}\n")

        # ---------- Adam ----------
        g = 2 * w_adam
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g * g

        bias1 = 1 - BETA1**t
        bias2 = 1 - BETA2**t
        m_hat = m / bias1
        v_hat = v / bias2
        adam_step = LR * m_hat / (math.sqrt(v_hat) + EPS)

        print(f"[Adam] 当前 w = {w_adam:.4f}，梯度 g = 2w = {g:.4f}")
        print(f"[Adam] 账本 m = 0.9×m_旧 + 0.1×g = {m:.6f}")
        print(f"[Adam] 账本 v = 0.999×v_旧 + 0.001×g² = {v:.8f}")
        print(f"[Adam] 修正 m̂ = m / (1-0.9^{t}) = {m:.6f} / {bias1:.3f} = {m_hat:.6f}")
        print(f"[Adam] 修正 v̂ = v / (1-0.999^{t}) = {v:.8f} / {bias2:.6f} = {v_hat:.6f}")
        print(
            f"[Adam] 步长 = lr × m̂ / √v̂ = 0.1 × {m_hat:.4f} / "
            f"{math.sqrt(v_hat):.4f} = {adam_step:.6f}"
        )
        w_adam -= adam_step
        print(f"[Adam] 新 w = {w_adam:.4f}，loss = {w_adam**2:.6f}\n")


if __name__ == "__main__":
    main()
