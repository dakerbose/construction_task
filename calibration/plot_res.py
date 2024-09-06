import matplotlib.pyplot as plt

# 读取txt文件
file_path = 'res.txt'
iterations = []
costs = []
gradients = []
steps = []
trust_radius = []
total_time = []
ls_iter = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('   '):  # 过滤掉非数据行
            parts = line.split()
            iterations.append(int(parts[0]))
            costs.append(float(parts[1]))
            gradients.append(float(parts[3]))
            steps.append(float(parts[4]))
            trust_radius.append(float(parts[6]))
            ls_iter.append(int(parts[7]))
            total_time.append(float(parts[9]))

# 绘制cost随迭代次数的变化图
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.plot(iterations, costs, marker='o', color='b', label='Cost')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Cost')
# plt.title('Cost vs Iteration')
plt.legend()

# 绘制gradient随迭代次数的变化图
plt.subplot(2, 3, 2)
plt.plot(iterations, gradients, marker='o', color='r', label='Gradient')
plt.xlabel('Iteration')
plt.yscale('log')
plt.ylabel('Gradient')
# plt.title('Gradient vs Iteration')
plt.legend()

# 绘制step随迭代次数的变化图
plt.subplot(2, 3, 3)
plt.plot(iterations, steps, marker='o', color='g', label='Step')
plt.xlabel('Iteration')
plt.ylabel('Step')
# plt.title('Step vs Iteration')
plt.legend()

# 绘制trust_radius随迭代次数的变化图
plt.subplot(2, 3, 4)
plt.plot(iterations, trust_radius, marker='o', color='y', label='trust_radius')
plt.xlabel('Iteration')
plt.ylabel('trust_radius')
plt.yscale('log')
# plt.title('trust_radius vs Iteration')
plt.legend()

# 绘制ls_iter随迭代次数的变化图
plt.subplot(2, 3, 5)
plt.plot(iterations, ls_iter, marker='o', color='k', label='ls_iter')
plt.xlabel('Iteration')
plt.ylabel('ls_iter')
# plt.title('ls_iter vs Iteration')
plt.legend()

# 绘制total_time随迭代次数的变化图
plt.subplot(2, 3, 6)
plt.plot(iterations, total_time, marker='o', color='m', label='total_time')
plt.xlabel('Iteration')
plt.ylabel('total_time')
# plt.title('total_time vs Iteration')
plt.legend()

plt.tight_layout()
plt.show()