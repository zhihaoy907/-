import PyDealII.Release as dealii
import sys
import time
import traceback
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
    QGauss2,
    PreconditionSSOR,
    SparsityPattern,
    ZeroFunction2,
    SolutionTransfer2,
    KellyErrorEstimator2,
    VectorTools2,
    FEValues2,
    FullMatrix,
    GeometryInfo2,
    ParameterHandler,
    Integer,
    Double,
    Point,
    Task,
    TaskGroup,
    BlockVector,
    Tensor1
)

update_values = 0x0001
update_gradients = 0x0002
update_quadrature_points = 0x0020
update_JxW_values = 0x0040

class MaterialData:
    def __init__(self, n_groups):
        self.n_groups = n_groups
        self.n_materials = 8

        self.diffusion = [[0.0] * self.n_groups for _ in range(self.n_materials)]
        self.sigma_r = [[0.0] * self.n_groups for _ in range(self.n_materials)]
        self.nu_sigma_f = [[0.0] * self.n_groups for _ in range(self.n_materials)]
        self.sigma_s = [[[0.0] * self.n_groups for _ in range(self.n_groups)] for _ in range(self.n_materials)]
        self.chi = [[0.0] * self.n_groups for _ in range(self.n_materials)]

        if self.n_groups == 2:
            for m in range(self.n_materials):
                self.diffusion[m][0] = 1.2
                self.diffusion[m][1] = 0.4
                self.chi[m][0] = 1.0
                self.chi[m][1] = 0.0
                self.sigma_r[m][0] = 0.03
                for group_1 in range(self.n_groups):
                    for group_2 in range(self.n_groups):
                        self.sigma_s[m][group_1][group_2] = 0.0

            self.diffusion[5][1] = 0.2

            self.sigma_r[4][0] = 0.026
            self.sigma_r[5][0] = 0.051
            self.sigma_r[6][0] = 0.026
            self.sigma_r[7][0] = 0.050

            self.sigma_r[0][1] = 0.100
            self.sigma_r[1][1] = 0.200
            self.sigma_r[2][1] = 0.250
            self.sigma_r[3][1] = 0.300
            self.sigma_r[4][1] = 0.020
            self.sigma_r[5][1] = 0.040
            self.sigma_r[6][1] = 0.020
            self.sigma_r[7][1] = 0.800

            self.nu_sigma_f[0][0] = 0.0050
            self.nu_sigma_f[1][0] = 0.0075
            self.nu_sigma_f[2][0] = 0.0075
            self.nu_sigma_f[3][0] = 0.0075
            self.nu_sigma_f[4][0] = 0.000
            self.nu_sigma_f[5][0] = 0.000
            self.nu_sigma_f[6][0] = 1e-7
            self.nu_sigma_f[7][0] = 0.00

            self.nu_sigma_f[0][1] = 0.125
            self.nu_sigma_f[1][1] = 0.300
            self.nu_sigma_f[2][1] = 0.375
            self.nu_sigma_f[3][1] = 0.450
            self.nu_sigma_f[4][1] = 0.000
            self.nu_sigma_f[5][1] = 0.000
            self.nu_sigma_f[6][1] = 3e-6
            self.nu_sigma_f[7][1] = 0.00

            self.sigma_s[0][0][1] = 0.020
            self.sigma_s[1][0][1] = 0.015
            self.sigma_s[2][0][1] = 0.015
            self.sigma_s[3][0][1] = 0.015
            self.sigma_s[4][0][1] = 0.025
            self.sigma_s[5][0][1] = 0.050
            self.sigma_s[6][0][1] = 0.025
            self.sigma_s[7][0][1] = 0.010
        else:
            raise ValueError("Currently, only data for 2 groups is implemented")


    def get_diffusion_coefficient(self, group, material_id):
        return self.diffusion[material_id][group]

    def get_removal_XS(self, group, material_id):
        return self.sigma_r[material_id][group]

    def get_fission_XS(self, group, material_id):
        return self.nu_sigma_f[material_id][group]

    def get_scattering_XS(self, group_1, group_2, material_id):
        return self.sigma_s[material_id, group_1, group_2]

    def get_fission_spectrum(self, group, material_id):
        return self.chi[material_id][group]

    def get_fission_dist_XS(self, group_1, group_2, material_id):
        return (self.get_fission_spectrum(group_1, material_id) *
                self.get_fission_XS(group_2, material_id))



class EnergyGroup:
    def __init__(self, group, material_data, coarse_grid, fe, dim):
        self.group = group
        self.material_data = material_data
        self.fe = fe
        self.coarse_grid = coarse_grid
        self.dof_handler = DoFHandler2(coarse_grid)
        self.triangulation = coarse_grid
        self.dof_handler.distribute_dofs(fe)
        self.dim = dim
        self.solution = Vector()
        self.solution_old = Vector()
        self.hanging_node_constraints = AffineConstraints()
        self.system_matrix = SparseMatrix()
        self.sparsity_pattern = SparsityPattern()
        self.system_rhs = Vector()
        self.boundary_values = {}


    def n_active_cells(self):
        return self.triangulation.n_active_cells()

    def n_dofs(self):
        return self.dof_handler.n_dofs()

    def setup_linear_system(self):
        n_dofs = self.dof_handler.n_dofs()

        self.hanging_node_constraints.clear()
        dealii.make_hanging_node_constraints2(self.dof_handler, self.hanging_node_constraints)
        self.hanging_node_constraints.close()
        
        self.system_matrix.clear()
        dsp = DynamicSparsityPattern(n_dofs, n_dofs)
        dealii.make_sparsity_pattern2(self.dof_handler, dsp)
        self.hanging_node_constraints.condense(dsp)

        self.sparsity_pattern.copy_from(dsp)
        self.system_matrix.reinit(self.sparsity_pattern)
        self.system_rhs.reinit(n_dofs)
        
        if self.solution.size() == 0:
            self.solution.reinit(n_dofs)
            self.solution_old.reinit(n_dofs)
            self.solution_old[0] = 1.0 
            self.solution = self.solution_old
        
        self.boundary_values.clear()

        for i in range(self.dim):
            VectorTools2.interpolate_boundary_values(self.dof_handler, 2 * i + 1, ZeroFunction2(), self.boundary_values)
    
    def assemble_system_matrix(self):
        quadrature_formula = QGauss2(self.fe.degree + 1)
        flags = update_values | update_quadrature_points | update_JxW_values | update_gradients
        flags = dealii.UpdateFlags(flags)
        fe_values = FEValues2(self.fe, quadrature_formula, flags)

        dofs_per_cell = self.fe.n_dofs_per_cell()
        n_q_points = quadrature_formula.size()
        cell_matrix = FullMatrix(dofs_per_cell, dofs_per_cell)
        local_dof_indices = [0] * dofs_per_cell


        for cell in self.dof_handler.active_cell_iterators():
            cell_matrix = FullMatrix(dofs_per_cell, dofs_per_cell)
            fe_values.reinit(cell)

            diffusion_coefficient = self.material_data.get_diffusion_coefficient(self.group, cell.material_id)
            removal_XS = self.material_data.get_removal_XS(self.group, cell.material_id)

            for q_point in range(n_q_points):
                for i in range(dofs_per_cell):
                    for j in range(dofs_per_cell):
                        cell_matrix[i,j] += (
                            (diffusion_coefficient * fe_values.shape_grad(i, q_point) *
                            fe_values.shape_grad(j, q_point) +
                            removal_XS * fe_values.shape_value(i, q_point) *
                            fe_values.shape_value(j, q_point)) *
                            fe_values.JxW(q_point)
                        )

            cell.get_dof_indices(local_dof_indices)
            
            for i in range(dofs_per_cell):
                for j in range(dofs_per_cell):
                    self.system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j))

        self.hanging_node_constraints.condense(self.system_matrix)

    def assemble_ingroup_rhs(self, extraneous_source):
        self.system_rhs.reinit(self.dof_handler.n_dofs())

        quadrature_formula = QGauss2(self.fe.degree + 1)
        flags = update_values | update_quadrature_points | update_JxW_values | update_gradients
        flags = dealii.UpdateFlags(flags)
        fe_values = FEValues2(self.fe, quadrature_formula, flags)

        dofs_per_cell = self.fe.n_dofs_per_cell()
        n_q_points = quadrature_formula.size()
        
        cell_rhs = Vector(dofs_per_cell)
        extraneous_source_values = [0.0] * n_q_points
        solution_old_values = [0.0] * n_q_points
        local_dof_indices = [0] * dofs_per_cell

        for cell in self.dof_handler.active_cell_iterators():
            cell_rhs = 0
            fe_values.reinit(cell)

            fission_dist_XS = self.material_data.get_fission_dist_XS(self.group, self.group, cell.material_id)

            extraneous_source.value_list(fe_values.get_quadrature_points(), extraneous_source_values)
            fe_values.get_function_values(self.solution_old, solution_old_values)

            cell.get_dof_indices(local_dof_indices)

            for q_point in range(n_q_points):
                for i in range(dofs_per_cell):
                    cell_rhs[i] += (
                        (extraneous_source_values[q_point] + fission_dist_XS * solution_old_values[q_point]) *
                        fe_values.shape_value(i, q_point) * fe_values.JxW(q_point)
                    )

            for i in range(dofs_per_cell):
                self.system_rhs[local_dof_indices[i]] += cell_rhs(i)

    def assemble_cross_group_rhs(self, g_prime):
        if self.group == g_prime.group:
            return

        cell_list = GridTools.get_finest_common_cells(self.dof_handler, g_prime.dof_handler)

        for cell_pair in cell_list:
            unit_matrix = FullMatrix(self.fe.n_dofs_per_cell())
            unit_matrix.fill(1)
            self.assemble_cross_group_rhs_recursive(g_prime, cell_pair.first, cell_pair.second, unit_matrix)

    def assemble_cross_group_rhs_recursive(self, g_prime, cell_g, cell_g_prime, prolongation_matrix):
        if not cell_g.has_children() and not cell_g_prime.has_children():
            quadrature_formula = QGauss2(self.fe.degree + 1)
            n_q_points = quadrature_formula.size()

            flags = update_values | update_quadrature_points
            fe_values = FEValues2(self.fe, flags)

            if cell_g.level() > cell_g_prime.level():
                fe_values.reinit(cell_g)
            else:
                fe_values.reinit(cell_g_prime)

            fission_dist_XS = self.material_data.get_fission_dist_XS(self.group, g_prime.group, cell_g_prime.material_id)
            scattering_XS = self.material_data.get_scattering_XS(g_prime.group, self.group, cell_g_prime.material_id)

            local_mass_matrix_f = FullMatrix(self.fe.n_dofs_per_cell(), self.fe.n_dofs_per_cell())
            local_mass_matrix_g = FullMatrix(self.fe.n_dofs_per_cell(), self.fe.n_dofs_per_cell())

            for q_point in range(n_q_points):
                for i in range(self.fe.n_dofs_per_cell()):
                    for j in range(self.fe.n_dofs_per_cell()):
                        local_mass_matrix_f[i,j] += (
                            fission_dist_XS * fe_values.shape_value(i, q_point) *
                            fe_values.shape_value(j, q_point) * fe_values.JxW(q_point)
                        )
                        local_mass_matrix_g[i,j] += (
                            scattering_XS * fe_values.shape_value(i, q_point) *
                            fe_values.shape_value(j, q_point) * fe_values.JxW(q_point)
                        )

            g_prime_new_values = Vector(self.fe.n_dofs_per_cell())
            g_prime_old_values = Vector(self.fe.n_dofs_per_cell())
            cell_g_prime.get_dof_values(g_prime.solution_old, g_prime_old_values)
            cell_g_prime.get_dof_values(g_prime.solution, g_prime_new_values)

            cell_rhs = Vector(self.fe.n_dofs_per_cell())
            tmp = Vector(self.fe.n_dofs_per_cell())

            if cell_g.level() > cell_g_prime.level():
                prolongation_matrix.vmult(tmp, g_prime_old_values)
                local_mass_matrix_f.vmult(cell_rhs, tmp)

                prolongation_matrix.vmult(tmp, g_prime_new_values)
                local_mass_matrix_g.vmult_add(cell_rhs, tmp)
            else:
                local_mass_matrix_f.vmult(tmp, g_prime_old_values)
                prolongation_matrix.Tvmult(cell_rhs, tmp)

                local_mass_matrix_g.vmult(tmp, g_prime_new_values)
                prolongation_matrix.Tvmult_add(cell_rhs, tmp)

            local_dof_indices = [0] * self.fe.n_dofs_per_cell()
            cell_g.get_dof_indices(local_dof_indices)

            for i in range(self.fe.n_dofs_per_cell()):
                self.system_rhs[local_dof_indices[i]] += cell_rhs(i)

        else:
            for child in range(GeometryInfo2.max_children_per_cell):
                new_matrix = FullMatrix(self.fe.n_dofs_per_cell(), self.fe.n_dofs_per_cell())
                self.fe.get_prolongation_matrix(child).mmult(new_matrix, prolongation_matrix)

                if cell_g.has_child(child):
                    self.assemble_cross_group_rhs_recursive(g_prime, cell_g.child(child), cell_g_prime, new_matrix)

                elif cell_g_prime.has_child(child):
                    self.assemble_cross_group_rhs_recursive(g_prime, cell_g, cell_g_prime.child(child), new_matrix)

    def solve(self):
        preconditioner = PreconditionSSOR(self.system_matrix)
        solver = SolverCG(self.system_matrix)
        solver.solve(self.system_matrix, self.solution, self.system_rhs, preconditioner)

    def get_fission_source(self):
        fission_source = 0
        for i in range(self.dof_handler.n_dofs()):
            fission_source += self.solution(i)
        return fission_source
    def solve(self):
        self.hanging_node_constraints.condense(self.system_rhs)

        dealii.apply_boundary_values(self.boundary_values,
                                          self.system_matrix,
                                          self.solution,
                                          self.system_rhs)

        solver_control = SolverControl(self.system_matrix.m(), 1e-12 * self.system_rhs.l2_norm())
        cg = SolverCG(solver_control)

        preconditioner = PreconditionSSOR(self.system_matrix)
        preconditioner.initialize(self.system_matrix, 1.2)

        cg.solve(self.system_matrix, self.solution, self.system_rhs, preconditioner)

        self.hanging_node_constraints.distribute(self.solution)

    def estimate_errors(self, error_indicators):
        KellyErrorEstimator2.estimate(
            self.dof_handler,
            QGauss2(self.dim - 1, self.fe.degree + 1),
            {},
            self.solution,
            error_indicators
        )
        error_indicators /= self.solution.linfty_norm()

    def refine_grid(self, error_indicators, refine_threshold, coarsen_threshold):
        
        for cell in self.triangulation.active_cell_iterators():
            if error_indicators[cell.active_cell_index()] > refine_threshold:
                cell.set_refine_flag()
            elif error_indicators[cell.active_cell_index()] < coarsen_threshold:
                cell.set_coarsen_flag()

        soltrans = SolutionTransfer2(self.dof_handler)
        self.triangulation.prepare_coarsening_and_refinement()
        soltrans.prepare_for_coarsening_and_refinement(self.solution)

        self.triangulation.execute_coarsening_and_refinement()
        self.dof_handler.distribute_dofs(self.fe)

        self.setup_linear_system()

        self.solution.reinit(self.dof_handler.n_dofs())
        soltrans.interpolate(self.solution_old, self.solution)

        self.hanging_node_constraints.distribute(self.solution)

        self.solution_old.reinit(self.dof_handler.n_dofs())
        self.solution_old = self.solution

    def output_results(self, cycle):
        filename = f"solution-{str(self.group).zfill(2)}.{str(cycle).zfill(2)}.vtu"

        data_out = DataOut2(self.dim)
        data_out.attach_dof_handler(self.dof_handler)
        data_out.add_data_vector(self.solution, "solution")
        data_out.build_patches()

        with open(filename, 'w') as output:
            data_out.write_vtu(output)



class NeutronDiffusionProblem:
    class Parameters:
        def __init__(self):
            self.n_groups = 2
            self.n_refinement_cycles = 5
            self.fe_degree = 2
            self.convergence_tolerance = 1e-12

        @staticmethod
        def declare_parameters(prm):
            prm.declare_entry("Number of energy groups", "2", Integer(),
                              "The number of energy different groups considered")
            prm.declare_entry("Refinement cycles", "5", Integer(),
                              "Number of refinement cycles to be performed")
            prm.declare_entry("Finite element degree", "2", Integer(),
                              "Polynomial degree of the finite element to be used")
            prm.declare_entry("Power iteration tolerance", "1e-12", Double(),
                              "Inner power iterations are stopped when the change in k_eff falls below this tolerance")

        def get_parameters(self, prm):
            self.n_groups = prm.get_integer("Number of energy groups")
            self.n_refinement_cycles = prm.get_integer("Refinement cycles")
            self.fe_degree = prm.get_integer("Finite element degree")
            self.convergence_tolerance = prm.get_double("Power iteration tolerance")

    def __init__(self, parameters):
        self.parameters = parameters
        self.material_data = MaterialData(parameters.n_groups)
        self.fe = FE_Q2(parameters.fe_degree)
        self.k_eff = 1.0 
        self.energy_groups = []
        self.convergence_table_stream = open("convergence_table.txt", "w")

    def initialize_problem(self,dim):
        rods_per_assembly_x = 17
        rods_per_assembly_y = 17
        pin_pitch_x = 1.26
        pin_pitch_y = 1.26
        assembly_height = 200

        assemblies_x = 2
        assemblies_y = 2
        assemblies_z = 1

        bottom_left = Point([0, 0, 0])

        if dim == 2:
            upper_right = Point([assemblies_x * rods_per_assembly_x * pin_pitch_x,
                                    assemblies_y * rods_per_assembly_y * pin_pitch_y])
        elif dim == 3:
            upper_right = Point([assemblies_x * rods_per_assembly_x * pin_pitch_x,
                                    assemblies_y * rods_per_assembly_y * pin_pitch_y,
                                    assemblies_z * assembly_height])

        # n_subdivisions 计算
        n_subdivisions = [assemblies_x * rods_per_assembly_x]
        if dim >= 2:
            n_subdivisions.append(assemblies_y * rods_per_assembly_y)
        if dim >= 3:
            n_subdivisions.append(assemblies_z)

        coarse_grid = Triangulation("2D","2D")
        dealii.subdivided_hyper_rectangle2(coarse_grid, n_subdivisions, bottom_left, upper_right, True)
        print(f"514: grid cells: {coarse_grid.n_active_cells()}")
        # n_assemblies = 4
        assembly_materials = [
            [
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1],
                [1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 5, 1, 1, 5, 1, 1, 7, 1, 1, 5, 1, 1, 5, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1],
                [1, 1, 1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ]
            ],
            [
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 1, 1, 1],
                [1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 8, 1, 1, 8, 1, 1, 7, 1, 1, 8, 1, 1, 8, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1],
                [1, 1, 1, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ]
            ],
            [
                [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
                [2, 3, 3, 3, 3, 5, 3, 3, 5, 3, 3, 5, 3, 3, 3, 3, 2],
                [2, 3, 3, 5, 3, 4, 4, 4, 4, 4, 4, 4, 3, 5, 3, 3, 2],
                [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2],
                [2, 3, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 3, 2],
                [2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2],
                [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 2],
                [2, 3, 3, 5, 4, 4, 4, 4, 5, 4, 4, 5, 4, 4, 5, 3, 2],
                [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2],
                [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 2],
                [2, 3, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 3, 2],
                [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2],
                [2, 3, 4, 5, 4, 4, 5, 4, 5, 5, 5, 5, 3, 4, 5, 3, 2],
                [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2],
                [2, 3, 3, 3, 4, 4, 5, 4, 5, 5, 5, 5, 3, 3, 3, 3, 2]
                ]
            ],
            [
                [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2],
                [2, 2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2],
                [2, 2, 2, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2],
                [2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2],
                [2, 2, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 3, 2],
                [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2],
                [2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 2, 2],
                [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2],
                [2, 2, 2, 2, 3, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 2],
                [2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 2],
                [2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 3, 2],
                [2, 2, 3, 2, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3]
                ]
            ]
        ]

        core = [
        [[0], [2]],
        [[2], [0]]
        ]

        for cell in coarse_grid.active_cell_iterators():
            cell_center = cell.center()

            tmp_x = int(cell_center.x / pin_pitch_x)
            ax = tmp_x / rods_per_assembly_x
            cx = tmp_x - ax * rods_per_assembly_x

            tmp_y = int(cell_center.y / pin_pitch_y)
            ay = tmp_y / rods_per_assembly_y
            cy = tmp_y - ay * rods_per_assembly_y

            az = 0 if dim == 2 else int(cell_center[dim - 1] / assembly_height)

            ax_int = int(ax)
            ay_int = int(ay)
            az_int = int(az)
            cx_int = int(cx)
            cy_int = int(cy)

            cell.material_id = assembly_materials[core[ax_int][ay_int][az_int]][cx_int][cy_int][0] - 1

        # 初始化能量组
        for group in range(self.parameters.n_groups):
            energy_group = EnergyGroup(group, self.material_data, coarse_grid, self.fe, dim)
            self.energy_groups.append(energy_group)
    
    def get_total_fission_source(self):
        fission_sources = [0.0] * self.parameters.n_groups

        tasks = TaskGroup()
        for group in range(self.parameters.n_groups):
            task = Task(self.energy_groups[group].get_fission_source())
            tasks += task

        tasks.joinall()

        return sum(fission_sources)

    def refine_grid(self):
        n_cells = [0] * self.parameters.n_groups
        for group in range(self.parameters.n_groups):
            n_cells[group] = self.energy_groups[group].n_active_cells()

        group_error_indicators = BlockVector(n_cells)

        tasks = TaskGroup()
        for group in range(self.parameters.n_groups):
            task = Task(self.energy_groups[group].estimate_errors(group_error_indicators.block(group)))
            tasks += task

        tasks.join_all()

        max_error = group_error_indicators.linfty_norm()
        refine_threshold = 0.3 * max_error
        coarsen_threshold = 0.01 * max_error

        tasks = TaskGroup()
        for group in range(self.parameters.n_groups):
            task = Task(self.energy_groups[group].refine_grid(group_error_indicators.block(group),refine_threshold,coarsen_threshold))
            tasks += task

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = sys.__stdout__
        print(f"{'Cycle':<8}{'k_eff':<10}{'flux_ratio':<15}{'max_thermal':<15}")
        sys.stdout = old_stdout

        
        k_eff_old = 0.0
        
        for cycle in range(self.parameters.n_refinement_cycles):
            timer = time.time()
            
            print(f"Cycle {cycle}:")
            
            if cycle == 0:
                self.initialize_problem(2)
                print(f"active cells: {self.energy_groups[0].n_active_cells()}")
                for group in range(self.parameters.n_groups):
                    self.energy_groups[group].setup_linear_system()
            else:
                self.refine_grid()
                for group in range(self.parameters.n_groups):
                    self.energy_groups[group].solution *= self.k_eff
                    
            print("   Numbers of active cells:       ", end="")
            for group in range(self.parameters.n_groups):
                print(f"{self.energy_groups[group].n_active_cells()} ", end="")
            print()
            
            print("   Numbers of degrees of freedom: ", end="")
            for group in range(self.parameters.n_groups):
                print(f"{self.energy_groups[group].n_dofs()} ", end="")
            print("\n")
            
            # 创建并行任务组
            tasks = TaskGroup()
            for group in range(self.parameters.n_groups):
                task = Task(self.energy_groups[group].assemble_system_matrix())
                tasks += task

            tasks.join_all()
                    
            # 幂迭代
            iteration = 1
            while True:
                # 处理每个能群
                for group in range(self.parameters.n_groups):
                    # 组装本能群右端项
                    self.energy_groups[group].assemble_ingroup_rhs(
                        ZeroFunction(self.dim)
                    )
                    
                    # 组装跨能群耦合项
                    for bgroup in range(self.parameters.n_groups):
                        self.energy_groups[group].assemble_cross_group_rhs(
                            self.energy_groups[bgroup]
                        )
                        
                    # 求解线性系统
                    self.energy_groups[group].solve()
                    
                # 计算新的k_eff和误差
                k_eff = self.get_total_fission_source()
                error = abs(k_eff - k_eff_old) / abs(k_eff)
                
                # 计算通量比和最大热中子通量
                flux_ratio = (self.energy_groups[0].solution.linfty_norm() /
                            self.energy_groups[1].solution.linfty_norm())
                max_thermal = self.energy_groups[1].solution.linfty_norm()
                
                # 打印迭代信息
                print(f"Iter number:{iteration:>2d} k_eff={k_eff:.12f} "
                    f"flux_ratio={flux_ratio:.12f} "
                    f"max_thermal={max_thermal:.12f}")
                    
                k_eff_old = k_eff
                
                # 更新旧解
                for group in range(self.parameters.n_groups):
                    self.energy_groups[group].solution_old = \
                        self.energy_groups[group].solution.copy()
                    self.energy_groups[group].solution_old /= k_eff
                    
                iteration += 1
                
                # 检查收敛性
                if error <= self.parameters.convergence_tolerance or iteration >= 500:
                    break
                    
            # 输出收敛信息到文件
            self.convergence_table_stream.write(
                f"{cycle} {self.energy_groups[0].n_dofs()} "
                f"{self.energy_groups[1].n_dofs()} {k_eff} "
                f"{self.energy_groups[0].solution.linfty_norm() / self.energy_groups[1].solution.linfty_norm()}\n"
            )
            
            # 输出每个能群的结果
            for group in range(self.parameters.n_groups):
                self.energy_groups[group].output_results(cycle)
                
            # 打印循环信息和计时
            print()
            print(f"   Cycle={cycle}, n_dofs="
                f"{self.energy_groups[0].n_dofs() + self.energy_groups[1].n_dofs()}, "
                f"k_eff={k_eff}, time={time.time() - timer}")
            print("\n")

def main():
    try:
        filename = "project.prm"
        if len(sys.argv) > 1:
            filename = sys.argv[1]

        parameter_handler = ParameterHandler()

        parameters = NeutronDiffusionProblem.Parameters()
        parameters.declare_parameters(parameter_handler)

        parameter_handler.parse_input(filename)

        parameters.get_parameters(parameter_handler)

        neutron_diffusion_problem = NeutronDiffusionProblem(parameters)
        neutron_diffusion_problem.run()

    except Exception as exc:
        print("\n" + "-"*50)
        print(f"Exception occurred: {exc}")
        traceback.print_exc()
        print("Aborting!")
        print("-"*50)
        sys.exit(1)

    except:
        print("\n" + "-"*50)
        print("Unknown exception occurred!")
        print("Aborting!")
        print("-"*50)
        sys.exit(1)

if __name__ == "__main__":
    main()
