from __future__ import division
import os, sys
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import AlternativePerformances, PerformanceTable
import math

verbose = False

class MipMRSort():

    def __init__(self, model, pt, aa, epsilon = 0.01, version_mip = 2, mip_nb_threads = 1, mip_timetype = 1, mip_timeout = 60):
        self.pt = pt
        self.aa = aa
        self.model = model
        self.criteria = model.criteria.get_active()
        self.cps = model.categories_profiles
        self.version_mip = version_mip
        self.mip_nb_threads = mip_nb_threads
        self.mip_timetype = mip_timetype
        self.mip_timeout = mip_timeout

        self.epsilon = epsilon
        self.bigm = 2

        self.__profiles = self.cps.get_ordered_profiles()
        self.__categories = self.cps.get_ordered_categories()

        # transforming decreasing criteria to minus increasing criteria
        #import pdb; pdb.set_trace()
        self.pt.update_direction(model.criteria)   

        #checking if there was some (additionnal) entries of the performance table of self.model and adding to pt
        if self.model.bpt is not None:
            import pdb; pdb.set_trace()
            # print(self.model.bpt)
            #import pdb; pdb.set_trace()
            self.model.bpt.update_direction(model.criteria)

            tmp_pt = self.pt.copy()
            for bp in self.model.bpt:
                tmp_pt.append(bp)

            self.ap_min = tmp_pt.get_min()
            self.ap_max = tmp_pt.get_max()
            self.ap_range = tmp_pt.get_range()
        else:
            #import pdb; pdb.set_trace()
            self.ap_min = self.pt.get_min()
            self.ap_max = self.pt.get_max()
            self.ap_range = self.pt.get_range()
            #import pdb; pdb.set_trace()

        for c in self.criteria:
            self.ap_min.performances[c.id] -= self.epsilon
            self.ap_max.performances[c.id] += self.epsilon
            self.ap_range.performances[c.id] += 2 * self.epsilon * 100
        #import pdb; pdb.set_trace()
        
        solver = os.getenv('SOLVER', 'cplex')
        if solver == 'glpk':
            import pymprog
            self.lp = pymprog.model('lp_elecre_tri_weights')
            self.lp.verb = verbose
            self.add_variables_glpk()
            self.add_constraints_glpk()
            self.add_extra_constraints_glpk()
            self.add_objective_glpk()
            self.solve_function = self.solve_glpk
        elif solver == 'cplex':
            import cplex
            solver_max_threads = int(os.getenv('SOLVER_MAX_THREADS', 0))
            self.lp = cplex.Cplex()
            self.lp.parameters.threads.set(solver_max_threads)
            self.lp.parameters.simplex.tolerances.optimality.set(0.000000001)
            self.lp.parameters.simplex.tolerances.feasibility.set(0.000001)
            self.lp.parameters.simplex.tolerances.markowitz.set(0.001)
            self.lp.parameters.mip.tolerances.mipgap.set(0.00001)
            #self.lp.parameters.timelimit.set(3600)
            self.lp.parameters.clocktype.set(self.mip_timetype)
            if self.mip_nb_threads>=0:
                self.lp.parameters.threads.set(self.mip_nb_threads)
            if mip_timeout>=0:
                self.lp.parameters.timelimit.set(mip_timeout) # 1h timeout
            #print(self.lp.parameters.simplex.tolerances.optimality.get())
            #import pdb; pdb.set_trace()
            self.add_variables_cplex()
            self.add_constraints_cplex()
            self.add_extra_constraints_cplex()
            self.add_objective_cplex()
            self.solve_function = self.solve_cplex
            if verbose is False:
                self.lp.set_log_stream(None)
                self.lp.set_results_stream(None)
#                self.lp.set_warning_stream(None)
#                self.lp.set_error_stream(None)
        else:
            raise NameError('Invalid solver selected')

        # re-transforming minus increasing criteria to decreasing criteria 
        # here is a re-encoding according to original values of pt (just after using the desired pt in the constraints equations) 
        self.pt.update_direction(model.criteria)
        if self.model.bpt is not None:
            self.model.bpt.update_direction(model.criteria)

    # def add_variables_glpk(self):
    #     n = len(self.criteria)
    #     m = len(self.aa)
    #     ncat = len(self.__categories)
    #     a1 = self.aa.get_alternatives_in_categories(self.__categories[1:])
    #     a2 = self.aa.get_alternatives_in_categories(self.__categories[:-1])

    #     self.a = {}
    #     for a in self.aa:
    #         self.a[a.id] = self.lp.var(name = "a_%s" % a.id, kind = bool)

    #     self.w = {}
    #     for c in self.criteria:
    #         self.w[c.id] = self.lp.var(name = "w_%s" % c.id, bounds = (0, 1))

    #     self.lbda = self.lp.var(name = 'lambda', bounds = (0.5, 1))

    #     self.g = {p: {} for p in self.__profiles}
    #     for p, c in product(self.__profiles, self.criteria):
    #         self.g[p][c.id] = self.lp.var(name = "g_%s_%s" % (p, c.id),
    #                                       bounds = (self.ap_min.performances[c.id],
    #                                                 self.ap_max.performances[c.id]))

    #     self.cinf = {a: {} for a in a1}
    #     self.dinf = {a: {} for a in a1}
    #     for a, c in product(a1, self.criteria):
    #         self.cinf[a][c.id] = self.lp.var(name = "cinf_%s_%s" % (a, c.id),
    #                                          bounds = (0, 1))
    #         self.dinf[a][c.id] = self.lp.var(name = "dinf_%s_%s" % (a, c.id),
    #                                          kind = bool)

    #     self.csup = {a: {} for a in a2}
    #     self.dsup = {a: {} for a in a2}
    #     for a, c in product(a2, self.criteria):
    #         self.csup[a][c.id] = self.lp.var(name = "csup_%s_%s" % (a, c.id),
    #                                          bounds = (0, 1))
    #         self.dsup[a][c.id] = self.lp.var(name = "dsup_%s_%s" % (a, c.id),
    #                                          kind = bool)

    #definition of variable herre : modifications
    def add_variables_cplex(self):
        self.lp.variables.add(names = ["gamma_" + a for a in self.aa.keys()],
                               types = [self.lp.variables.type.binary
                                        for a in self.aa.keys()])
                              # lb = [-1000 for a in self.aa.keys()],
                              # ub = [10000 for a in self.aa.keys()])
        self.lp.variables.add(names = ["lambda"], lb = [0], ub = [1])
        self.lp.variables.add(names = ["w_" + c.id for c in self.criteria],
                              lb = [0 for c in self.criteria],
                              ub = [1 for c in self.criteria])
    
        self.lp.variables.add(names = ["b_m_"+ c.id for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["g_%s_%s" % (profile, c.id)
                                       for profile in self.__profiles
                                       for c in self.criteria],
                              lb = [self.ap_min.performances[c.id] if abs(c.direction)==1 else self.epsilon
                                    for profile in self.__profiles
                                    for c in self.criteria],
                              ub = [self.ap_max.performances[c.id] + self.epsilon if abs(c.direction)==1 else 0.5
                                    for profile in self.__profiles
                                    for c in self.criteria])
        #print([self.ap_min.performances[c.id] if abs(c.direction)==1 else self.epsilon for profile in self.__profiles for c in self.criteria])
        #print([self.ap_max.performances[c.id] + self.epsilon if abs(c.direction)==1 else 0.5-self.epsilon for profile in self.__profiles for c in self.criteria])
        #import pdb; pdb.set_trace()
        #additionnal variable (b^*) for single peaked criterions (A DEFINIR UNIQUEMENT SUR LE DERNIER CRITERE ICI QUI EST SINGLE PEAKED)
        # note thtat (b_*) is defined before = "g_%s_%s"
        #import pdb; pdb.set_trace()
        # self.lp.variables.add(names = ["g2_%s_%s" % (profile, c.id)
        #                                for profile in self.__profiles
        #                                for c in self.criteria],
        #                       lb = [self.ap_min.performances[c.id]
        #                             for profile in self.__profiles
        #                             for c in self.criteria],
        #                       ub = [self.ap_max.performances[c.id] + self.epsilon
        #                             for profile in self.__profiles
        #                             for c in self.criteria])
        #import pdb; pdb.set_trace()
        
        a1 = self.aa.get_alternatives_in_categories(self.__categories[1:])
        a2 = self.aa.get_alternatives_in_categories(self.__categories[:-1])
        self.lp.variables.add(names = ["cinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria],
                              lb = [0 for a in a1 for c in self.criteria],
                              ub = [1 for a in a1 for c in self.criteria])
        self.lp.variables.add(names = ["csup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria],
                              lb = [0 for a in a2 for c in self.criteria],
                              ub = [1 for a in a2 for c in self.criteria])
        self.lp.variables.add(names = ["dinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria],
                              types = [self.lp.variables.type.binary
                                       for a in a1
                                       for c in self.criteria])
        self.lp.variables.add(names = ["dsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria],
                              types = [self.lp.variables.type.binary
                                       for a in a2
                                       for c in self.criteria]) 
        
        self.lp.variables.add(names = ["betainf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              types = [self.lp.variables.type.binary
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["betasup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              types = [self.lp.variables.type.binary
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["sigma_%s" % (c.id)
                                       for c in self.criteria if c.direction==0],
                              types = [self.lp.variables.type.binary
                                       for c in self.criteria if c.direction==0]) 
        
        # self.lp.variables.add(names = ["alphainf_%s_%s" % (a, c.id)
        #                                for a in a1
        #                                for c in self.criteria],
        #                       lb = [0 for a in a1 for c in self.criteria],
        #                       ub = [1 for a in a1 for c in self.criteria])
        # self.lp.variables.add(names = ["alphasup_%s_%s" % (a, c.id)
        #                                for a in a2
        #                                for c in self.criteria],
        #                       lb = [0 for a in a2 for c in self.criteria],
        #                       ub = [1 for a in a2 for c in self.criteria])
        self.lp.variables.add(names = ["alphapinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["alphapsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["alphaminf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a1 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        self.lp.variables.add(names = ["alphamsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              lb = [0 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0],
                              ub = [1 for a in a2 for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        #import pdb; pdb.set_trace()
        #ajout criteria singlepeaked =>delta^* car delta_* est deja definie (c'est le delta dans la cas de critere non SP)
        # self.lp.variables.add(names = ["dinf2_%s_%s" % (a, c.id)
        #                                for a in a1
        #                                for c in self.criteria],
        #                       types = [self.lp.variables.type.binary
        #                                for a in a1
        #                                for c in self.criteria])
        # self.lp.variables.add(names = ["dsup2_%s_%s" % (a, c.id)
        #                                for a in a2
        #                                for c in self.criteria],
        #                       types = [self.lp.variables.type.binary
        #                                for a in a2
        #                                for c in self.criteria])
        #import pdb; pdb.set_trace()


    # constraints additionnal with single peaked criterion..
    def __add_lower_constraints_cplex(self, aa):
        constraints = self.lp.linear_constraints
        i = self.__categories.index(aa.category_id)
        b = self.__profiles[i - 1]
        bigm = self.bigm
        
        # sum cinf_j(a_i, b_{h-1}) >= lambda - 2 (1 - alpha_i)
        # sum cinf_j(a_i, b_{h-1}) >= lambda + M (alpha_i - 1)
        constraints.add(names = ["gamma_inf_%s" % (aa.id)],
                        lin_expr =
                            [
                              [["cinf_%s_%s" % (aa.id, c.id) for c in self.criteria] + ["lambda"] + ["gamma_%s" % aa.id],
                              [1 for c in self.criteria] + [-1] + [-bigm]],
                            ],
                        senses = ["G"],
                        rhs = [-bigm]
                        )       
        
        # constraints.add(names = ["gamma_inf_%s_ahdoc" % (aa.id)],
        #                 lin_expr =
        #                     [
        #                       [["a_%s" % aa.id],
        #                       [1]],
        #                     ],
        #                 senses = ["E"],
        #                 rhs = [1]
        #                 )        
        # constraints.add(names = ["gamma_inf_%s" % (aa.id)],
        #                 lin_expr =
        #                     [
        #                      [["cinf_%s_%s" % (aa.id, c.id)
        #                        for c in self.criteria] + \
        #                       ["lambda"] + ["a_%s" % aa.id],
        #                       [1 for c in self.criteria] + [-1] + [-1]],
        #                     ],
        #                 senses = ["E"],
        #                 rhs = [0]
        #                )
        #import pdb; pdb.set_trace()

        for c in self.criteria:
            bigm = self.ap_range.performances[c.id]
            bigm = self.bigm
            #print("bigm inf",bigm)

            # cinf_j(a_i, b_{h-1}) <= w_j
            constraints.add(names = ["c_cinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["cinf_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )

            
            # cinf_j(a_i, b_{h-1}) <= dinf_{i,j}
            constraints.add(names = ["c_cinf2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["cinf_%s_%s" % (aa.id, c.id), "dinf_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            
            # cinf_j(a_i, b_{h-1}) >= dinf_{i,j} - 1 + w_j
            constraints.add(names = ["c_cinf3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["cinf_%s_%s" % (aa.id, c.id), "dinf_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                            )

            if abs(c.direction) == 1:
                # M dinf_(i,j) > a_{i,j} - b_{h-1,j}
                # M dinf_(i,j) >= a_{i,j} - b_{h-1,j} + epsilon
                constraints.add(names = ["d_dinf1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["G"],
                                rhs = [self.pt[aa.id].performances[c.id] + self.epsilon]
                                )
    
                
                # M dinf_(i,j) <= a_{i,j} - b_{h-1,j} + M
                # M (dinf_(i,j) - 1) <= a_{i,j} - b_{h-1,j}
                constraints.add(names = ["d_dinf2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [self.pt[aa.id].performances[c.id] + bigm]
                                )
                
                
            # additionnal constraints single peaked criteria
             # cinf_j(a_i, b_{h-1}) <= dinf_{i,j} + dinf2_{i,j} - 1
            #import pdb; pdb.set_trace()
            #if c in self.criteria[-1]:
            if abs(c.direction) == 2 or c.direction == 0:
                
                #3eme formulation
                ##   alphapinf_(i,j) - alphaminf_(i,j) = b_m - a_{i,j}
                constraints.add(names = ["alphapminf_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id), "b_m_%s" % (c.id)],
                                      [1, -1, -1]],
                                    ],
                                senses = ["E"],
                                rhs = [-self.pt[aa.id].performances[c.id]]
                                )
                
                constraints.add(names = ["alphapminf1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapinf_%s_%s" % (aa.id, c.id), "betainf_%s_%s" % (aa.id, c.id)],
                                      [1, -bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [0]
                                )
                constraints.add(names = ["alphapminf2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphaminf_%s_%s" % (aa.id, c.id), "betainf_%s_%s" % (aa.id, c.id)],
                                      [1, bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [bigm]
                                )
                
                # bm + b_* <= 1
                constraints.add(names = ["bmbinf1_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [1]
                                )
                # bm - b_* >= 0
                constraints.add(names = ["bmbinf2_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, -1]],
                                    ],
                                senses = ["G"],
                                rhs = [0]
                                )
                
                if c.direction==0:
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## M(1 - sigma_i) + M dinf_(i,j) > b_{h-1,j} - alphapinf_{i,j} - alphaminf_{i,j} => SP
                    constraints.add(names = ["d_dinf3_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon - bigm]
                                    )
        
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## M(sigma - 1) + M (dinf_(i,j)-1) <= b_{h-1,j} - alphapinf_{i,j} - alphaminf_{i,j} => SP
                    constraints.add(names = ["d_dinf4_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm + bigm]
                                    )
                    
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## M sigma_i + M dinf_(i,j)  > alphapinf_{i,j} + alphaminf_{i,j} - b_{h-1,j} => SV
                    constraints.add(names = ["d_dinf5_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
        
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## -M sigma_i + M (dinf_(i,j)-1) <= alphapinf_{i,j} + alphaminf_{i,j} - b_{h-1,j} => SV
                    constraints.add(names = ["d_dinf6_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                
                if c.direction == 2:
                    # M dinf_(i,j) > a_{i,j} - b_{h-1,j}
                    # M dinf_(i,j) >= alphinf_{i,j} - b_{h-1,j} + epsilon
                    # alpha*(-1) since SP peaked criteria became cost criterion, hence opposite evaluations are to be maximize 
                    # (2eme formulation) : M dinf_(i,j) > b_{h-1,j} - alpha_{i,j}
                    # 3eme formulation avec "alphapinf_i_j et alphaminf_i_j
                    constraints.add(names = ["d_dinf13_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
        
                    # M dinf_(i,j) <= a_{i,j} - b_{h-1,j} + M
                    # M (dinf_(i,j) - 1) <= alphinf_{i,j} - b_{h-1,j}
                    # alpha*(-1) since SP peaked criteria became cost criterion, hence opposite evaluations are to be maximize
                    # (2eme formulation) : M (dinf_(i,j)-1) <= b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dinf14_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                    
                if c.direction == -2:
                    # M dinf_(i,j) > a_{i,j} - b_{h-1,j}
                    # M dinf_(i,j) >= alphinf_{i,j} - b_{h-1,j} + epsilon (since single valley=> that becomes a increasing criterion)
                    constraints.add(names = ["d_dinf23_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
                    # M dinf_(i,j) <= a_{i,j} - b_{h-1,j} + M
                    # M (dinf_(i,j) - 1) <= alphinf_{i,j} - b_{h-1,j}
                    constraints.add(names = ["d_dinf24_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dinf_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapinf_%s_%s" % (aa.id, c.id), "alphaminf_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                #print(constraints.get_rhs(),constraints.get_senses(),constraints.get_names(),constraints.get_rows())
                #import pdb; pdb.set_trace()
            
            
    # suite constraint single peaked criteria
    def __add_upper_constraints_cplex(self, aa):
        constraints = self.lp.linear_constraints
        i = self.__categories.index(aa.category_id)
        b = self.__profiles[i]
        bigm = self.bigm
        
        # sum csup_j(a_i, b_{h-1}) < lambda + 2 (1 - gamma_i)
        # sum csup_j(a_i, b_{h-1}) + epsilon <= lambda - M (gamma_i - 1)
        constraints.add(names = ["gamma_sup_%s" % (aa.id)],
                        lin_expr =
                            [
                              [["csup_%s_%s" % (aa.id, c.id) for c in self.criteria] + ["lambda"] + ["gamma_%s" % aa.id],
                              [1 for c in self.criteria] + [-1] + [bigm]],
                            ],
                        senses = ["L"],
                        rhs = [bigm - self.epsilon]
                        )
        
        # constraints.add(names = ["gamma_sup_%s_ahdoc" % (aa.id)],
        #                 lin_expr =
        #                     [
        #                       [["a_%s" % aa.id],
        #                       [1]],
        #                     ],
        #                 senses = ["E"],
        #                 rhs = [1]
        #                 )
        # constraints.add(names = ["gamma_sup_%s" % (aa.id)],
        #                 lin_expr =
        #                     [
        #                      [["csup_%s_%s" % (aa.id, c.id)
        #                        for c in self.criteria] + \
        #                       ["lambda"] + ["a_%s" % aa.id],
        #                       [1 for c in self.criteria] + [-1] + [1]],
        #                     ],
        #                 senses = ["E"],
        #                 rhs = [0 - self.epsilon]
        #                )
            
        for c in self.criteria:
            bigm = self.ap_range.performances[c.id]
            bigm = self.bigm
            #print("bigm",bigm)

            # csup_j(a_i, b_h) <= w_j
            constraints.add(names = ["c_csup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["csup_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            # csup_j(a_i, b_h) <= dsup_{i,j}
            constraints.add(names = ["c_csup2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["csup_%s_%s" % (aa.id, c.id), "dsup_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                            )
            
            # csup_j(a_i, b_{h-1}) >= dsup_{i,j} - 1 + w_j
            constraints.add(names = ["c_csup3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                  [["csup_%s_%s" % (aa.id, c.id), "dsup_%s_%s" % (aa.id, c.id), "w_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                            )
            
            if abs(c.direction) == 1:
                # M dsup_(i,j) >= a_{i,j} - b_{h-1,j} + epsilon
                constraints.add(names = ["d_dsup1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["G"],
                                rhs = [self.pt[aa.id].performances[c.id] + self.epsilon]
                                )
    
                
                # M (dsup_(i,j) - 1) <= a_{i,j} - b_{h-1,j}
                constraints.add(names = ["d_dsup2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id)],
                                      [bigm, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [self.pt[aa.id].performances[c.id] + bigm]
                                )
            
            
            # additionnal constraints single peaked criteria (IN ADDITION)
            # csup_j(a_i, b_{h-1}) <= dsup_{i,j} + dsup2_{i,j} - 1
            #import pdb; pdb.set_trace()
            #if c in self.criteria[-1]:
            if c.direction == 0 or abs(c.direction)==2:
                                            
                #3eme formulation
                ##   alphapsup_(i,j) - alphaminf_(i,j) = b_m - a_{i,j}
                constraints.add(names = ["alphapmsup_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id), "b_m_%s" % (c.id)],
                                      [1, -1, -1]],
                                    ],
                                senses = ["E"],
                                rhs = [-self.pt[aa.id].performances[c.id]]
                                )
                
                constraints.add(names = ["alphapmsup1_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphapsup_%s_%s" % (aa.id, c.id), "betasup_%s_%s" % (aa.id, c.id)],
                                      [1, -bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [0]
                                )
                constraints.add(names = ["alphapmsup2_%s_%s" % (aa.id, c.id)],
                                lin_expr =
                                    [
                                      [["alphamsup_%s_%s" % (aa.id, c.id), "betasup_%s_%s" % (aa.id, c.id)],
                                      [1, bigm]],
                                    ],
                                senses = ["L"],
                                rhs = [bigm]
                                )
    
                #bm + b_* <= 1
                constraints.add(names = ["bmbsup1_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, 1]],
                                    ],
                                senses = ["L"],
                                rhs = [1]
                                )
                
                # bm - b_* >= 0
                constraints.add(names = ["bmbsup2_%s_%s" % (b, c.id)],
                                lin_expr =
                                    [
                                      [["b_m_%s" % (c.id) , "g_%s_%s" % (b, c.id)],
                                      [1, -1]],
                                    ],
                                senses = ["G"],
                                rhs = [0]
                                )   
                
                if c.direction ==0:
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## M(1 - sigma_i) + M dsup_(i,j) > b_{h-1,j} - alphapsup_{i,j} - alphamsup_{i,j} =====> SP
                    constraints.add(names = ["d_dsup3_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon - bigm]
                                    )
        
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## M(sigma - 1) + M (dsup_(i,j) - 1) <= b_{h-1,j} - alphapsup_{i,j} - alphamsup_{i,j} ====> SP
                    constraints.add(names = ["d_dsup4_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm + bigm]
                                    )
                    
                    # grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    # M sigma_i + M dsup_(i,j)  > alphapsup_{i,j} + alphamsup_{i,j} - b_{h-1,j} ====> SV
                    constraints.add(names = ["d_dsup5_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
        
                    ## grosses contraintes : decomposition de la valeur absolue + var binaire pour determiner SP ou SV
                    ## -M sigma_i + M (dsup_(i,j)-1) <= alphapsup_{i,j} + alphamsup_{i,j} - b_{h-1,j} ====> SV
                    constraints.add(names = ["d_dsup6_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "sigma_%s" % (c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
             
                if c.direction == 2:
                    # M dsup_(i,j) >= alphsup_{i,j} - b_{h-1,j} + epsilon
                    # (1rst formulmation) alpha*(-1) since SP peaked criteria became cost criterion, hence opposite evaluations are to be maximize
                    # (2eme formulation : it integrates the fact that decreasing criterion means <b_i is good)
                    # (2eme formulation) : M dsup_(i,j) > b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dsup13_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
                    # M (dsup_(i,j) - 1) <= alphsup_{i,j} - b_{h-1,j}
                    # (1rst formulmation) alpha*(-1) since SP peaked criteria became cost criterion, hence opposite evaluations are to be maximize
                    # (2eme formulation : it integrates the fact that decreasing criterion means <b_i is good)
                    # (2eme formulation) : M (dsup_(i,j)-1) <= b_{h-1,j} - alpha_{i,j}
                    constraints.add(names = ["d_dsup14_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, -1, 1, 1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                
                if c.direction == -2:
                    # M dsup_(i,j) >= alphsup_{i,j} - b_{h-1,j} + epsilon (single valley turns to an increasing criterion)
                    constraints.add(names = ["d_dsup23_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["G"],
                                    rhs = [self.epsilon]
                                    )
                    # M (dsup_(i,j) - 1) <= alphsup_{i,j} - b_{h-1,j}  (single valley turns to an increasing criterion)
                    constraints.add(names = ["d_dsup24_%s_%s" % (aa.id, c.id)],
                                    lin_expr =
                                        [
                                          [["dsup_%s_%s" % (aa.id, c.id), "g_%s_%s" % (b, c.id), "alphapsup_%s_%s" % (aa.id, c.id), "alphamsup_%s_%s" % (aa.id, c.id)],
                                          [bigm, 1, -1, -1]],
                                        ],
                                    senses = ["L"],
                                    rhs = [bigm]
                                    )
                #print(constraints.get_rhs(),constraints.get_senses(),constraints.get_names(),constraints.get_rows())
                #import pdb; pdb.set_trace()



    def add_alternatives_constraints(self):
        lower_cat = self.__categories[0]
        upper_cat = self.__categories[-1]

        for aa in self.aa:
            cat = aa.category_id

            if cat != lower_cat:
                self.add_lower_constraints(aa)

            if cat != upper_cat:
                self.add_upper_constraints(aa)

    #usual constraints...
    def add_constraints_cplex(self):
        constraints = self.lp.linear_constraints

        self.add_lower_constraints = self.__add_lower_constraints_cplex
        self.add_upper_constraints = self.__add_upper_constraints_cplex
        self.add_alternatives_constraints()

        profiles = self.cps.get_ordered_profiles()
        for h, c in product(range(len(profiles) - 1), self.criteria):
            # g_j(b_h) <= g_j(b_{h+1})
            print("dominance")
            constraints.add(names= ["dominance"],
                            lin_expr =
                                [
                                 [["g_%s_%s" % (profiles[h], c.id),
                                   "g_%s_%s" % (profiles[h + 1], c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

        # sum w_j = 1
        if self.model.cv is None:
            constraints.add(names = ["wsum"],
                            lin_expr =
                                [
                                  [["w_%s" % c.id for c in self.criteria],
                                  [1 for c in self.criteria]],
                                ],
                            senses = ["E"],
                            rhs = [1]
                            )

    #these constrains fixed values that are already defined in the model
    def add_extra_constraints_cplex(self):
        constraints = self.lp.linear_constraints
        #print(self.model.lbda,self.model.cv,self.model.bpt)
        
        #add some additionnal constraints
        #import pdb; pdb.set_trace()
        
        # if self.model.lbda is not None:
        #     constraints.add(names = ["lambda"],
        #                     lin_expr =
        #                         [
        #                          [["lambda"],
        #                           [1]]
        #                          ],
        #                     senses = ["E"],
        #                     rhs = [self.model.lbda]
        #                    )

        # constraints.add(names = ["d_dinf_sigma"],
        #                 lin_expr =
        #                     [
        #                       [["sigma_c1"],
        #                       [1]],
        #                     ],
        #                 senses = ["E"],
        #                 rhs = [1]
        #                 )
        
        # constraints.add(names = ["lambda"],
        #                 lin_expr =
        #                     [
        #                       [["lambda"],
        #                       [1]]
        #                       ],
        #                 senses = ["E"],
        #                 rhs = [0.6]
        #                     )
            
        # constraints.add(names = ["b_m_c3"],
        #                 lin_expr =
        #                     [
        #                       [["b_m_c3"],
        #                       [1]]
        #                       ],
        #                 senses = ["E"],
        #                 rhs = [0.225]
        #                     )

        # if self.model.cv is not None:
        #     for c in self.criteria:
        #         constraints.add(names = ["w_%s" % c.id],
        #                         lin_expr =
        #                             [
        #                              [["w_%s" % c.id],
        #                               [1]]
        #                              ],
        #                         senses = ["E"],
        #                         rhs = [self.model.cv[c.id].value]
        #                        )

        # for i,j in [("c1",0.3),("c2",0.3),("c3",0.4)]:
        #     constraints.add(names = ["w_%s" % i],
        #                     lin_expr =
        #                         [
        #                           [["w_%s" % i],
        #                           [1]]
        #                           ],
        #                     senses = ["E"],
        #                     rhs = [j]
        #                     )
            
        # if self.model.bpt is not None:
        #     for bp, c in product(self.model.bpt, self.model.criteria):
        #         constraints.add(names = ["g_%s_%s" % (bp.id, c.id)],
        #                         lin_expr =
        #                             [
        #                              [["g_%s_%s" % (bp.id, c.id)],
        #                               [1]]
        #                              ],
        #                         senses = ["E"],
        #                         rhs = [bp.performances[c.id]]
        #                        )
        
        #for i,j in [("c1",0.3),("c2",0.5),("c3",0.7)]:
        # for i,j in [("c1",0.9)]:
        #     constraints.add(names = ["g_%s_%s" % ("b1", i)],
        #                     lin_expr =
        #                         [
        #                           [["g_%s_%s" % ("b1", i)],
        #                           [1]]
        #                           ],
        #                     senses = ["E"],
        #                     rhs = [j]
        #                     )
            
        #import pdb; pdb.set_trace()
                
                
    def add_objective_cplex(self):
        a1 = self.aa.get_alternatives_in_categories(self.__categories[1:])
        a2 = self.aa.get_alternatives_in_categories(self.__categories[:-1])

        lex = 10**(int(math.log10(len(self.aa)))+1)
        eps = round(1./len(self.criteria),3)
        # print([("g_%s_%s" % (profile, c.id), eps) for profile in self.__profiles for c in self.criteria if abs(c.direction)==0])
        #print([("gamma_%s" % aid, 1) for aid in self.aa.keys()])
        #print([[("alphainf_%s_%s" % (a, c.id),-1) for a in a1 for c in self.criteria]])
        #import pdb; pdb.set_trace()
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)
        #self.lp.objective.set_linear([("gamma_%s" % aid, 1) for aid in self.aa.keys()])
        # self.lp.objective.set_linear([("gamma_%s" % aid, 1) for aid in self.aa.keys()] +
        #                              [("g_%s_%s" % (profile, c.id), eps) for profile in self.__profiles for c in self.criteria if abs(c.direction)==0])
        if self.version_mip == 2:
            self.lp.objective.set_linear([("gamma_%s" % aid, len(self.criteria)+1) for aid in self.aa.keys()] + [("sigma_%s" % (c.id), 1) for c in self.criteria if abs(c.direction)==0] + [("g_%s_%s" % (profile, c.id), eps) for profile in self.__profiles for c in self.criteria if abs(c.direction)==0])
        else :
            self.lp.objective.set_linear([("gamma_%s" % aid, 1) for aid in self.aa.keys()])
                                     # [("zalpha", (-1)*100)] +
                                     #    [("alphainf_%s_%s" % (a, c.id),(-1)*100) for a in a1 for c in self.criteria] +
                                     #    [("alphasup_%s_%s" % (a, c.id),(-1)*100) for a in a2 for c in self.criteria])
        # self.lp.objective.set_linear([("gamma_%s" % aid, 1) for aid in self.aa.keys()] + 
        #                                 [("alphainf_%s_%s" % (a, c.id),-4) for a in a1 for c in self.criteria] +
        #                                 [("alphasup_%s_%s" % (a, c.id),-4) for a in a2 for c in self.criteria])

    #main function
    def solve_cplex(self):
        #print("deb")
        import cplex
        cplex.Cplex().set_results_stream(None)
        cplex.Cplex().set_log_stream(None)
        start = cplex.Cplex().get_time()
        self.lp.solve()
        end = cplex.Cplex().get_time()
        cplex_time = end-start
        #print(cplex_time)
        status = self.lp.solution.get_status()
        # print(status)
        #print(status)
        #print(self.lp.parameters.emphasis.numerical)
        #import pdb; pdb.set_trace()
        
        if status != self.lp.solution.status.MIP_optimal and status != 102 and status != 107:
            raise RuntimeError("Solver status: %s" % status)

        obj = self.lp.solution.get_objective_value()

        cvs = CriteriaValues()
        for c in self.criteria:
            cv = CriterionValue()
            cv.id = c.id
            cv.value = self.lp.solution.get_values('w_' + c.id)
            cvs.append(cv)

        self.model.cv = cvs

        self.model.lbda = self.lp.solution.get_values("lambda")
        #print(self.model.cv, self.model.lbda )
        a1 = self.aa.get_alternatives_in_categories(self.__categories[1:])
        a2 = self.aa.get_alternatives_in_categories(self.__categories[:-1])
        #print(a1,a2)

        # print("obj", obj)
        # print("SP-SV sigma", [self.lp.solution.get_values("sigma_%s" % (c.id)) for c in self.criteria if abs(c.direction)==0])
        # print("sum SP profiles", [self.lp.solution.get_values("g_%s_%s" % (profile, c.id)) for profile in self.__profiles for c in self.criteria], sum([self.lp.solution.get_values("g_%s_%s" % (profile, c.id)) for profile in self.__profiles for c in self.criteria if abs(c.direction)==0]))
        # print("relative optimality gap", self.lp.solution.MIP.get_mip_relative_gap())
        # #print("zalpha",self.lp.solution.get_values("zalpha"))
        # print("gamma_", [self.lp.solution.get_values("gamma_" + a) for a in self.aa.keys()], sum([self.lp.solution.get_values("gamma_" + a) for a in self.aa.keys()]))
        # print("lbda", self.lp.solution.get_values("lambda"))
        # print("b_m", [self.lp.solution.get_values("b_m_%s" % (c.id)) for c in self.criteria if abs(c.direction)==2 or c.direction==0])
        # print("cinf", a1, [self.lp.solution.get_values("cinf_%s_%s" % (a, c.id)) for a in a1 for c in self.criteria])
        # print("csup", a2, [self.lp.solution.get_values("csup_%s_%s" % (a, c.id)) for a in a2 for c in self.criteria])
        #print("inf_delta", a1, [self.lp.solution.get_values("dinf_%s_%s" % (a, c.id)) for a in a1 for c in self.criteria])
        #print("sup_delta", a2, [self.lp.solution.get_values("dsup_%s_%s" % (a, c.id)) for a in a2 for c in self.criteria])
        # print("inf_delta", a1, [self.lp.solution.get_values("dinf_%s_%s" % ("pta69", c.id)) for c in self.criteria])
        # print("alpha+inf_ sp", a1, [self.lp.solution.get_values("alphapinf_%s_%s" % (a, c.id)) for a in a1 for c in self.criteria if abs(c.direction)==0])
        # print("alpha-inf_ sp", a1, [self.lp.solution.get_values("alphaminf_%s_%s" % (a, c.id)) for a in a1 for c in self.criteria if abs(c.direction)==0])
        # print("alphasup_ sp", a2, [self.lp.solution.get_values("alphapsup_%s_%s" % (a, c.id)) for a in a2 for c in self.criteria if abs(c.direction)==0])
        # print("alphasup_ sp", a2, [self.lp.solution.get_values("alphamsup_%s_%s" % (a, c.id)) for a in a2 for c in self.criteria if abs(c.direction)==0])
        #import pdb; pdb.set_trace()
        
        #Verify that each alternative is correctly assigned 
        # a1: cat2
        # lbda=self.lp.solution.get_values("lambda")
        # print("cat2")
        # for a in a1:
        #     if a=="pta69":
        #         #print(lbda,self.lp.solution.get_values("gamma_" + a))
        #         print([self.lp.solution.get_values("cinf_%s_%s" % (a, c.id)) for c in self.criteria])
        #         print(a,sum([self.lp.solution.get_values("cinf_%s_%s" % (a, c.id)) for c in self.criteria]),">=",lbda+(1000*(self.lp.solution.get_values("gamma_" + a)-1)))
        # print("cat1")
        # for a in a2:
        #     #print(lbda,self.lp.solution.get_values("gamma_" + a))
        #     print([self.lp.solution.get_values("csup_%s_%s" % (a, c.id)) for c in self.criteria])
        #     print(a,sum([self.lp.solution.get_values("csup_%s_%s" % (a, c.id)) for c in self.criteria])+0.001,"<=",lbda-(1000*(self.lp.solution.get_values("gamma_" + a)-1)))
        # #import pdb; pdb.set_trace() 
                
        pt = PerformanceTable()
        pt2 = PerformanceTable()
        for p in self.__profiles:
            ap = AlternativePerformances(p)
            ap2 = AlternativePerformances(p)
            for c in self.criteria:
                #import pdb; pdb.set_trace()
                perf = self.lp.solution.get_values("g_%s_%s" % (p, c.id))
                #print(c,perf)
                ap.performances[c.id] = round(perf, 5)
                ap2.performances[c.id] = round(perf, 5)
                # if abs(c.direction) == 1:
                #     ap2.performances[c.id] = round(perf, 5)
                if abs(c.direction)==2 or c.direction==0:
                    bm = self.lp.solution.get_values("b_m_%s" % (c.id))
                    ap.performances[c.id] = bm - round(perf, 5)
                    ap2.performances[c.id] = (bm - round(perf, 5),bm + round(perf, 5))
                    #perf2 = self.lp.solution.get_values("g2_%s_%s" % (p, c.id))
                    #print(ap.performances[c.id],bm,round(perf, 5))
                    #self.model.b_peak = self.lp.solution.get_values("b_m_%s" % (c.id))
                    #b_m = self.lp.solution.get_values("b_m")*(-1)
                    #ap2.performances[c.id] = (-round(perf, 5),b_m*2 - round(perf2, 5))
                    #ap2.performances[c.id] = (round(perf, 5),round(perf2, 5))
                #import pdb; pdb.set_trace()
                #print(c,perf,perf2)
            pt.append(ap)
            pt2.append(ap2)

        self.model.bpt = pt
        self.model.bpt_sp = pt2
        #self.model.b_peak = self.lp.solution.get_values("b_m")
        #self.model.b_peak = self.lp.solution.get_values("b_m")*(-1)
        #self.model.bpt_sp = 
        #self.model.bpt2 = pt2
        #print(self.model.bpt)
        self.model.bpt.update_direction(self.model.criteria)
        # for crit in self.model.criteria:
        #     if crit.direction ==2:
        #         for cat_i in self.model.bpt.keys():
        #             import pdb; pdb.set_trace()
        #             self.model2.bpt[cat_i].performances[crit.id] *= (-1)
        
        self.model.bpt_sp.update_direction(self.model.criteria)
        #self.model.bpt2.update_direction(self.model.criteria)

        # for i,j in self.model.criteria.items():
        #     if abs(self.model.criteria[i].direction) == 2:
        #         for cat_i in self.model.bpt.keys():
        #             self.model.bpt_sp[cat_i] = (self.model.bpt[cat_i].performances[i],pt2[cat_i].performances[i])

        #import pdb; pdb.set_trace()
        #print(self.model.bpt)

        #return obj,pt2
        #tmp1 = [self.lp.solution.get_values("alphainf_%s_%s" % (a, c.id)) for a in a1 for c in self.criteria if abs(c.direction)==2]+[self.lp.solution.get_values("alphasup_%s_%s" % (a, c.id)) for a in a2 for c in self.criteria if abs(c.direction)==2]
        #print([abs(self.lp.solution.get_values("b_m_%s" % (c.id)) - self.pt[a].performances[c.id]) for a in a1 for c in self.criteria if abs(c.direction)==2])
        #import pdb; pdb.set_trace()
        #tmp2 = [abs(self.lp.solution.get_values("b_m_%s" % (c.id)) - self.pt[a].performances[c.id]) for a in a1 for c in self.criteria if abs(c.direction)==2]+[abs(self.lp.solution.get_values("b_m_%s" % (c.id)) - self.pt[a].performances[c.id]) for a in a2 for c in self.criteria if abs(c.direction)==2]
        # print(tmp1)
        # print(tmp2)
        #import pdb; pdb.set_trace()
        # print([self.lp.solution.get_values("betainf_%s_%s" % (a, c.id)) for a in a1 for c in self.criteria if abs(c.direction)==2]+[self.lp.solution.get_values("betasup_%s_%s" % (a, c.id)) for a in a2 for c in self.criteria if abs(c.direction)==2])
        #tmp1 = [min(self.lp.solution.get_values("alphapinf_%s_%s" % (a, c.id)),self.lp.solution.get_values("alphaminf_%s_%s" % (a, c.id))) for a in a1 for c in self.criteria if abs(c.direction)==2]+[min(self.lp.solution.get_values("alphapsup_%s_%s" % (a, c.id)),self.lp.solution.get_values("alphamsup_%s_%s" % (a, c.id))) for a in a2 for c in self.criteria if abs(c.direction)==2]
        # print([(self.lp.solution.get_values("alphapinf_%s_%s" % (a, c.id)),self.lp.solution.get_values("alphaminf_%s_%s" % (a, c.id))) for a in a1 for c in self.criteria if abs(c.direction)==2]+[(self.lp.solution.get_values("alphapsup_%s_%s" % (a, c.id)),self.lp.solution.get_values("alphamsup_%s_%s" % (a, c.id))) for a in a2 for c in self.criteria if abs(c.direction)==2])
        tmp3 = [self.lp.solution.get_values("g_%s_%s" % (profile, c.id)) for profile in self.__profiles for c in self.criteria if c.direction==0]
        tmp4 = [(c.id, self.lp.solution.get_values("sigma_%s" % (c.id))) for c in self.criteria if abs(c.direction)==0]
        
        tmp5 = [self.lp.solution.get_values("sigma_%s" % (c.id)) for c in self.criteria if abs(c.direction)==0]
        tmp6 = [self.lp.solution.get_values("b_m_%s" % (c.id)) for c in self.criteria if c.direction==0]
        tmp7 = [self.lp.solution.get_values("g_%s_%s" % (profile, c.id)) for profile in self.__profiles for c in self.criteria if c.direction==0]
        # print(tmp5,tmp6,tmp7)
        return (obj,self.lp.solution.MIP.get_mip_relative_gap(),sum([self.lp.solution.get_values("gamma_" + a) for a in self.aa.keys()]), sum(tmp3),tmp4,tmp5,tmp6,tmp7, status, cplex_time)

        #return (obj,sum([self.lp.solution.get_values("gamma_" + a) for a in self.aa.keys()]),sum([abs(i-j) for i,j in zip(tmp1,tmp2)]))

    # def solve_glpk(self):
    #     self.lp.solvopt(method='exact', integer='advanced')
    #     self.lp.solve()

    #     status = self.lp.status()
    #     if status != 'opt':
    #         raise RuntimeError("Solver status: %s" % self.lp.status())

    #     #print(self.lp.reportKKT())
    #     obj = self.lp.vobj()

    #     cvs = CriteriaValues()
    #     for c in self.criteria:
    #         cv = CriterionValue()
    #         cv.id = c.id
    #         cv.value = float(self.w[c.id].primal)
    #         cvs.append(cv)

    #     self.model.cv = cvs
    #     self.model.lbda = self.lbda.primal

    #     pt = PerformanceTable()
    #     for p in self.__profiles:
    #         ap = AlternativePerformances(p)
    #         for c in self.criteria:
    #             perf = self.g[p][c.id].primal
    #             ap.performances[c.id] = round(perf, 5)
    #         pt.append(ap)

    #     self.model.bpt = pt
    #     self.model.bpt.update_direction(self.model.criteria)

    #     return obj

    def solve(self):
        return self.solve_function()

if __name__ == "__main__":
    from pymcda.generate import generate_random_mrsort_model
    from pymcda.generate import generate_alternatives
    from pymcda.generate import generate_random_performance_table
    from pymcda.utils import print_pt_and_assignments
    from pymcda.ui.graphic import display_electre_tri_models

    seed = 12
    ncrit = 5
    ncat = 3

    # Generate a random ELECTRE TRI BM model
    model = generate_random_mrsort_model(ncrit, ncat, seed)

    # Display model parameters
    print('Original model')
    print('==============')
    cids = model.criteria.keys()
    cids.sort()
    model.bpt.display(criterion_ids = cids)
    model.cv.display(criterion_ids = cids)
    print("lambda: %.7s" % model.lbda)

    # Generate a set of alternatives
    a = generate_alternatives(100)
    pt = generate_random_performance_table(a, model.criteria)

    worst = pt.get_worst(model.criteria)
    best = pt.get_best(model.criteria)

    # Assign the alternatives
    aa = model.pessimist(pt)

    # Run the MIP
    model2 = model.copy()
    model2.cv = model.cv
    model2.bpt = None
    model2.lbda = model.lbda

    mip = MipMRSort(model2, pt, aa)
    mip.solve()

    # Display learned model parameters
    print('Learned model')
    print('=============')
    model2.bpt.display(criterion_ids = cids)
    model2.cv.display(criterion_ids = cids)
    print("lambda: %.7s" % model2.lbda)

    # Compute assignment with the learned model
    aa2 = model2.pessimist(pt)

    # Compute CA
    total = len(a)
    nok = 0
    anok = []
    for alt in a:
        if aa(alt.id) != aa2(alt.id):
            anok.append(alt)
            nok += 1

    print("Good assignments: %3g %%" % (float(total-nok)/total*100))
    print("Bad assignments : %3g %%" % (float(nok)/total*100))

    if len(anok) > 0:
        print("Alternatives wrongly assigned:")
        print_pt_and_assignments(anok.keys(), model.criteria.keys(),
                                 [aa, aa2], pt)

    # Display models
    display_electre_tri_models([model, model2],
                               [worst, worst], [best, best])
