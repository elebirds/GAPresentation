import pygad
import numpy as np

# 定义背包问题的参数
items_weights = [4, 2, 3, 1, 2]  # 物品的重量
items_values = [10, 4, 7, 2, 5]  # 物品的价值
max_weight = 7  # 背包的最大承重
def fitness_func(ga_instance, solution, solution_idx):
    weight = np.sum(solution * items_weights)
    value = np.sum(solution * items_values)
    if weight > max_weight:
        return 0
    return value

# 定义遗传算法的参数
ga_instance = pygad.GA(
    num_generations=100, 
    num_parents_mating=5, 
    fitness_func=fitness_func,
    sol_per_pop=10, 
    num_genes=len(items_weights),
    gene_type=int,
    init_range_low=0,
    init_range_high=2,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

# 运行遗传算法
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"最佳解决方案: {solution}")
print(f"最佳解决方案的适应度: {solution_fitness}")
print(f"最佳解决方案的总重量: {np.sum(solution * items_weights)}")
print(f"最佳解决方案的总价值: {np.sum(solution * items_values)}")