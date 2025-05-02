import os
import jax
import jax.numpy as jnp

# 强制 JAX 使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设备检查
print("JAX Backend:", jax.default_backend())
print("Available devices:", jax.devices())

# 创建一个简单的矩阵计算
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)

print("Matrix multiplication successful!")

