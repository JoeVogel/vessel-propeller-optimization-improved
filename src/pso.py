import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class PSO():

    def __init__(self, obj_func, dimension=3, lower_bounds=None, upper_bounds=None, num_particles=30, w=0.5, c1=1, c2=2, seed=None):
        self.obj_function   = obj_func
        self.dim            = dimension
        self.num_particles  = num_particles 
        self.w              = w
        self.c1             = c1
        self.c2             = c2
        self.lower_bounds   = lower_bounds
        self.upper_bounds   = upper_bounds

        if seed is not None:
            np.random.seed(seed)

        # Initialize particles and velocities
        self.particles = np.array([np.random.uniform(self.lower_bounds[i], self.upper_bounds[i], self.num_particles) for i in range(self.dim)]).T
        self.velocities = np.zeros((self.num_particles, self.dim))

        # Initialize the best positions and fitness values
        self.best_positions = np.copy(self.particles)
        self.best_fitness = np.array([self.obj_function(p) for p in self.particles])
        self.swarm_best_position = self.best_positions[np.argmin(self.best_fitness)]
        self.swarm_best_fitness = np.min(self.best_fitness)

    def solve(self):
        # Update velocities
        r1 = np.random.uniform(size=(self.num_particles, self.dim))
        r2 = np.random.uniform(size=(self.num_particles, self.dim))
        self.velocities = self.w * self.velocities + self.c1 * r1 * (self.best_positions - self.particles) + self.c2 * r2 * (self.swarm_best_position - self.particles)

        # Update positions
        self.particles += self.velocities

        # Constrain the limits
        self.particles = np.clip(self.particles, self.lower_bounds, self.upper_bounds)

        # Evaluate fitness of each particle
        fitness_values = np.array([self.obj_function(p) for p in self.particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < self.best_fitness)
        self.best_positions[improved_indices] = self.particles[improved_indices]
        self.best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < self.swarm_best_fitness:
            self.swarm_best_position = self.particles[np.argmin(fitness_values)]
            self.swarm_best_fitness = np.min(fitness_values)

    def get_result(self):
        # Return the best solution found by the PSO algorithm
        return self.swarm_best_position, self.swarm_best_fitness

# # Define the dimensions of the problem
# dim = 2

# # Run the PSO algorithm on the Rastrigin function
# solution, fitness = pso(rastrigin, dim=dim)

# # Print the solution and fitness value
# print('Solution:', solution)
# print('Fitness:', fitness)

# # Create a meshgrid for visualization
# x = np.linspace(-5.12, 5.12, 100)
# y = np.linspace(-5.12, 5.12, 100)
# X, Y = np.meshgrid(x, y)
# Z = rastrigin([X, Y])

# # Create a 3D plot of the Rastrigin function
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# # Plot the solution found by the PSO algorithm
# ax.scatter(solution[0], solution[1], fitness, color='red')
# plt.show()