import PyDealII.Release as dealii
from PyDealII.Release import (
    Triangulation,
    FE_Q2,
    DoFHandler2,
    SolverControl,
    SolverCG,
    SparseMatrix,
    DynamicSparsityPattern,
    Vector,
    AffineConstraints,
    DataOut2,
    QGauss1,
    QGauss2,
    PreconditionSSOR,
    SparsityPattern,
    RightHandSide2,
    BoundaryValues2,
    ZeroFunction2,
    SolutionTransfer2,
    KellyErrorEstimator2,
    VectorTools2,
)

class HeatEquation:
    def __init__(self, dim=3):
        self.dim = dim
        self.triangulation = Triangulation("2D","2D")
        self.fe = FE_Q2(1)
        self.dof_handler = DoFHandler2(self.triangulation)
        self.constraints = AffineConstraints()
        self.sparsity_pattern = SparsityPattern()
        self.mass_matrix = SparseMatrix()
        self.laplace_matrix = SparseMatrix()
        self.system_matrix = SparseMatrix()
        self.solution = Vector()
        self.old_solution = Vector()
        self.system_rhs = Vector()
        self.time = 0.0
        self.time_step = 1.0 / 500.0
        self.timestep_number = 0
        self.theta = 0.5

    def setup_system(self):
        self.dof_handler.distribute_dofs(self.fe)
        print("\n===========================================")
        print("Number of active cells:", self.triangulation.n_active_cells())
        print("Number of degrees of freedom:", self.dof_handler.n_dofs())
        self.constraints.clear()
        dealii.make_hanging_node_constraints2(self.dof_handler, self.constraints)
        self.constraints.close()
        dsp = DynamicSparsityPattern(self.dof_handler.n_dofs())
        dealii.make_sparsity_pattern2(self.dof_handler, dsp, self.constraints)
        self.sparsity_pattern.copy_from(dsp)
        self.mass_matrix.reinit(self.sparsity_pattern)
        self.laplace_matrix.reinit(self.sparsity_pattern)
        self.system_matrix.reinit(self.sparsity_pattern)
        
        dealii.create_mass_matrix2(self.dof_handler, QGauss2(self.fe.degree + 1), self.mass_matrix)
        dealii.create_laplace_matrix2(self.dof_handler, QGauss2(self.fe.degree + 1), self.laplace_matrix)
        self.solution.reinit(self.dof_handler.n_dofs())
        self.old_solution.reinit(self.dof_handler.n_dofs())
        self.system_rhs.reinit(self.dof_handler.n_dofs())
 
    def solve_time_step(self):
        solver_control = SolverControl(1000, 1e-8 * self.system_rhs.l2_norm())
        cg = SolverCG(solver_control)

        preconditioner = PreconditionSSOR()
        preconditioner.initialize(self.system_matrix, 1.0)
            
        cg.solve(self.system_matrix, self.solution, self.system_rhs, preconditioner)
        self.constraints.distribute(self.solution)

        print(f"     {solver_control.last_step()} CG iterations.")

    def output_results(self):
        data_out = DataOut2()
        data_out.attach_dof_handler(self.dof_handler)
        data_out.add_data_vector(self.solution, "U")
        data_out.build_patches()
        data_out.set_flags(self.time, self.timestep_number)

        filename = f"solution-{str(self.timestep_number).zfill(3)}.vtk"
        with open(filename, 'w') as output:
            data_out.write_vtk(output)

    def refine_mesh(self, min_grid_level, max_grid_level):
        estimated_error_per_cell = Vector(self.triangulation.n_active_cells())
        
        KellyErrorEstimator2.estimate(
            self.dof_handler,
            QGauss1(self.fe.degree + 1), 
            {},  
            self.solution,
            estimated_error_per_cell 
        )
        
        dealii.refine_and_coarsen_fixed_fraction2(
            self.triangulation,
            estimated_error_per_cell,
            0.6, 
            0.4 
        )

        if self.triangulation.n_levels() > max_grid_level:
            for cell in self.triangulation.active_cell_iterators_on_level(max_grid_level):
                cell.clear_refine_flag()

        for cell in self.triangulation.active_cell_iterators_on_level(min_grid_level):
            cell.clear_coarsen_flag()

        solution_trans = SolutionTransfer2(self.dof_handler)
        previous_solution = Vector(self.solution)

        self.triangulation.prepare_coarsening_and_refinement()
        solution_trans.prepare_for_coarsening_and_refinement(previous_solution)
        self.triangulation.execute_coarsening_and_refinement()
        self.setup_system()
        solution_trans.interpolate(previous_solution, self.solution)
        self.constraints.distribute(self.solution)
    def run(self):
        initial_global_refinement = 2
        n_adaptive_pre_refinement_steps = 4

        dealii.hyper_L2(self.triangulation, -1, 1, False)
        self.triangulation.refine_global(initial_global_refinement)

        self.setup_system()
        
        pre_refinement_step = 0
        
        tmp = Vector(self.solution.size())
        forcing_terms = Vector(self.solution.size())
        
        

        while True:
            self.time = 0.0
            self.timestep_number = 0
            
            tmp = Vector(forcing_terms)
            forcing_terms.reinit(self.solution.size())

            n_components = 1
            VectorTools2.interpolate(self.dof_handler,dealii.ZeroFunction2(n_components), self.old_solution)
            self.solution = self.old_solution
            
            self.output_results()

            end_time = 0.5

            while self.time <= end_time:
                
                self.time += self.time_step
                self.timestep_number += 1
                
                print(f"Time step {self.timestep_number} at t={self.time}")

                self.mass_matrix.vmult(self.system_rhs, self.old_solution)
                self.laplace_matrix.vmult(tmp, self.old_solution)
                self.system_rhs.add(-(1 - self.theta) * self.time_step, tmp)
                
                rhs_function = dealii.RightHandSide2()
                rhs_function.set_time(self.time)
                

                dealii.VectorTools2.create_right_hand_side(
                    self.dof_handler,
                    QGauss2(self.fe.degree + 1),
                    rhs_function,
                    tmp
                )
                
                forcing_terms = Vector(tmp)
                forcing_terms *= (self.time_step * self.theta)           
                rhs_function.set_time(self.time - self.time_step)

                dealii.VectorTools2.create_right_hand_side(
                    self.dof_handler,
                    QGauss2(self.fe.degree + 1),
                    rhs_function,
                    tmp
                )

                forcing_terms.add(self.time_step * (1 - self.theta), tmp)
                self.system_rhs += forcing_terms
                self.system_matrix.copy_from(self.mass_matrix)
                self.system_matrix.add(self.theta * self.time_step, self.laplace_matrix)
                self.constraints.condense(self.system_matrix, self.system_rhs)

                boundary_values_function = BoundaryValues2()
                boundary_values_function.set_time(self.time)
                boundary_values = {}
                dealii.VectorTools2.interpolate_boundary_values(self.dof_handler, 0, boundary_values_function, boundary_values)
                dealii.apply_boundary_values(boundary_values, self.system_matrix, self.solution, self.system_rhs)

                self.solve_time_step()

                self.output_results()

                if self.timestep_number == 1 and pre_refinement_step < n_adaptive_pre_refinement_steps:
                    self.refine_mesh(
                        initial_global_refinement,
                        initial_global_refinement + n_adaptive_pre_refinement_steps
                    )
                    pre_refinement_step += 1
                    self.time = 0.0
                    self.timestep_number = 0
                    
                    tmp.reinit(self.solution.size())
                    tmp = Vector(self.solution)
                    forcing_terms.reinit(self.solution.size())

                    n_components = 1
                    VectorTools2.interpolate(self.dof_handler,dealii.ZeroFunction2(n_components), self.old_solution)
                    self.solution = self.old_solution
                    
                    self.output_results()
                    continue

                

                if self.timestep_number > 0 and self.timestep_number % 5 == 0:
                    self.refine_mesh(
                        initial_global_refinement,
                        initial_global_refinement + n_adaptive_pre_refinement_steps
                    )
                    tmp.reinit(self.solution.size())
                    tmp = Vector(self.solution)
                    forcing_terms.reinit(self.solution.size())
                    forcing_terms = Vector(forcing_terms)
                self.old_solution = self.solution

            if pre_refinement_step >= n_adaptive_pre_refinement_steps:
                break




if __name__ == "__main__":
    try:
        heat_equation_solver = HeatEquation(dim=2)
        heat_equation_solver.run()
    except Exception as exc:
        print("\n----------------------------------------------------")
        print("Exception on processing:", exc)
        print("Aborting!")
        print("----------------------------------------------------")
    except:
        print("\n----------------------------------------------------")
        print("Unknown exception!")
        print("Aborting!")
        print("----------------------------------------------------")
