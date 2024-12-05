#include <glpk.h>
#include <iostream>

int main() {
    // Create problem object
    glp_prob *lp = glp_create_prob();
    glp_set_prob_name(lp, "sample");
    glp_set_obj_dir(lp, GLP_MAX); // We are maximizing

    // Define number of variables and constraints
    glp_add_rows(lp, 3); // 3 constraints
    glp_set_row_name(lp, 1, "Constraint1");
    glp_set_row_bnds(lp, 1, GLP_UP, 0.0, 100.0); // x1 + x2 + x3 <= 100
    glp_set_row_name(lp, 2, "Constraint2");
    glp_set_row_bnds(lp, 2, GLP_UP, 0.0, 600.0); // 10x1 + 4x2 + 5x3 <= 600
    glp_set_row_name(lp, 3, "Constraint3");
    glp_set_row_bnds(lp, 3, GLP_UP, 0.0, 300.0); // 2x1 + 2x2 + 6x3 <= 300

    glp_add_cols(lp, 3); // 3 variables
    glp_set_col_name(lp, 1, "x1");
    glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0); // x1 >= 0
    glp_set_obj_coef(lp, 1, 10.0);             // Objective coefficient for x1

    glp_set_col_name(lp, 2, "x2");
    glp_set_col_bnds(lp, 2, GLP_LO, 0.0, 0.0); // x2 >= 0
    glp_set_obj_coef(lp, 2, 6.0);              // Objective coefficient for x2

    glp_set_col_name(lp, 3, "x3");
    glp_set_col_bnds(lp, 3, GLP_LO, 0.0, 0.0); // x3 >= 0
    glp_set_obj_coef(lp, 3, 4.0);              // Objective coefficient for x3

    // Define matrix for constraints
    int ia[1 + 9], ja[1 + 9];
    double ar[1 + 9];
    ia[1] = 1; ja[1] = 1; ar[1] = 1.0;  // Constraint1: +1x1
    ia[2] = 1; ja[2] = 2; ar[2] = 1.0;  // Constraint1: +1x2
    ia[3] = 1; ja[3] = 3; ar[3] = 1.0;  // Constraint1: +1x3
    ia[4] = 2; ja[4] = 1; ar[4] = 10.0; // Constraint2: +10x1
    ia[5] = 2; ja[5] = 2; ar[5] = 4.0;  // Constraint2: +4x2
    ia[6] = 2; ja[6] = 3; ar[6] = 5.0;  // Constraint2: +5x3
    ia[7] = 3; ja[7] = 1; ar[7] = 2.0;  // Constraint3: +2x1
    ia[8] = 3; ja[8] = 2; ar[8] = 2.0;  // Constraint3: +2x2
    ia[9] = 3; ja[9] = 3; ar[9] = 6.0;  // Constraint3: +6x3

    glp_load_matrix(lp, 9, ia, ja, ar);

    // Solve the problem
    glp_simplex(lp, NULL);

    // Output results
    double z = glp_get_obj_val(lp);
    std::cout << "Optimal value (z): " << z << std::endl;

    for (int i = 1; i <= 3; ++i) {
        std::cout << glp_get_col_name(lp, i) << " = " << glp_get_col_prim(lp, i) << std::endl;
    }

    // Clean up
    glp_delete_prob(lp);
    glp_free_env();

    return 0;
}
