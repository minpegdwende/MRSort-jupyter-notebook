from __future__ import division
import os, sys
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import AlternativePerformances, PerformanceTable

verbose = False

class MipMRSortVC():

    def __init__(self, model, pt, aa, indep_veto_weights = True,
                 epsilon = 0.0001):
        self.pt = pt
        self.aa = aa
        self.model = model
        self.criteria = model.criteria
        self.cps = model.categories_profiles

        self.indep_veto_weights = indep_veto_weights
        self.epsilon = epsilon

        self.__profiles = self.cps.get_ordered_profiles()
        self.__categories = self.cps.get_ordered_categories()

        solver = os.getenv('SOLVER', 'cplex')
        if solver == 'cplex':
            import cplex
            solver_max_threads = int(os.getenv('SOLVER_MAX_THREADS', 0))
            self.lp = cplex.Cplex()
            self.lp.parameters.threads.set(solver_max_threads)
            if verbose is False:
                self.lp.set_log_stream(None)
                self.lp.set_results_stream(None)
#                self.lp.set_warning_stream(None)
#                self.lp.set_error_stream(None)
        else:
            raise NameError('Invalid solver selected')

        self.pt.update_direction(model.criteria)
        self.add_variables()
        self.add_constraints()
        self.add_extra_constraints()
        self.add_objective()
        self.pt.update_direction(model.criteria)

    def add_variables(self):
        self.ap_min = self.pt.get_min()
        self.ap_max = self.pt.get_max()
        self.ap_range = self.pt.get_range()
        for c in self.criteria:
            self.ap_min.performances[c.id] -= self.epsilon
            self.ap_max.performances[c.id] += self.epsilon
            self.ap_range.performances[c.id] *= 2

        self.lp.variables.add(names = ["a_" + a for a in self.aa.keys()],
                              types = [self.lp.variables.type.binary
                                       for a in self.aa.keys()])
        self.lp.variables.add(names = ["lambda"], lb = [0.5], ub = [1])
        self.lp.variables.add(names = ["LAMBDA"], lb = [0.0], ub = [1.1])
        self.lp.variables.add(names = ["w_" + c.id for c in self.criteria],
                              lb = [0 for c in self.criteria],
                              ub = [1 for c in self.criteria])
        self.lp.variables.add(names = ["z_" + c.id for c in self.criteria],
                              lb = [0 for c in self.criteria],
                              ub = [1 for c in self.criteria])
        self.lp.variables.add(names = ["g_%s_%s" % (profile, c.id)
                                       for profile in self.__profiles
                                       for c in self.criteria],
                              lb = [self.ap_min.performances[c.id]
                                    for profile in self.__profiles
                                    for c in self.criteria],
                              ub = [self.ap_max.performances[c.id] + self.epsilon
                                    for profile in self.__profiles
                                    for c in self.criteria])
        self.lp.variables.add(names = ["v_%s_%s" % (profile, c.id)
                                       for profile in self.__profiles
                                       for c in self.criteria],
                              lb = [0
                                    for profile in self.__profiles
                                    for c in self.criteria],
                              ub = [self.ap_range.performances[c.id]
                                    for profile in self.__profiles
                                    for c in self.criteria])

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
        self.lp.variables.add(names = ["uinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria],
                              lb = [0 for a in a1 for c in self.criteria],
                              ub = [1 for a in a1 for c in self.criteria])
        self.lp.variables.add(names = ["usup_%s_%s" % (a, c.id)
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
        self.lp.variables.add(names = ["vinf_%s_%s" % (a, c.id)
                                       for a in a1
                                       for c in self.criteria],
                              types = [self.lp.variables.type.binary
                                       for a in a1
                                       for c in self.criteria])
        self.lp.variables.add(names = ["vsup_%s_%s" % (a, c.id)
                                       for a in a2
                                       for c in self.criteria],
                              types = [self.lp.variables.type.binary
                                       for a in a2
                                       for c in self.criteria])
        self.lp.variables.add(names = ["oinf_%s" % (a)
                                       for a in a1],
                              types = [self.lp.variables.type.binary
                                       for a in a1])
        self.lp.variables.add(names = ["osup_%s" % (a)
                                       for a in a2],
                              types = [self.lp.variables.type.binary
                                       for a in a2])

    def __add_alternative_lower_constraints(self, aa):
        constraints = self.lp.linear_constraints
        i = self.__categories.index(aa.category_id)
        b = self.__profiles[i - 1]

        # sum cinf_j(a_i, b_{h-1}) - oinf_a >= lambda - 2 (1 - alpha_i)
        constraints.add(names = ["gamma_inf_%s" % (aa.id)],
                        lin_expr =
                            [
                             [["cinf_%s_%s" % (aa.id, c.id)
                               for c in self.criteria] + \
                              ["oinf_%s" % (aa.id)] + \
                              ["lambda"] + ["a_%s" % aa.id],
                              [1 for c in self.criteria] + [-1] + [-1] + \
                              [-2]],
                            ],
                        senses = ["G"],
                        rhs = [-2]
                       )

        # sum uinf(a,j) - LAMBDA >= M (oinf_a - 1)
        constraints.add(names = ["veto_inf_%s" % (aa.id)],
                        lin_expr =
                            [
                             [["uinf_%s_%s" % (aa.id, c.id)
                               for c in self.criteria] + \
                              ["LAMBDA"] + ["oinf_%s" % (aa.id)],
                              [1 for c in self.criteria] + [-1] + [-2]],
                            ],
                        senses = ["G"],
                        rhs = [-2]
                       )

        # sum uinf(a,j) - LAMBDA < M oinf_a
        constraints.add(names = ["veto_inf2_%s" % (aa.id)],
                        lin_expr =
                            [
                             [["uinf_%s_%s" % (aa.id, c.id)
                               for c in self.criteria] + \
                              ["LAMBDA"] + ["oinf_%s" % (aa.id)],
                              [1 for c in self.criteria] + [-1] + [-2]],
                            ],
                        senses = ["L"],
                        rhs = [- self.epsilon]
                       )

        for c in self.criteria:
            bigm = self.ap_range.performances[c.id]

            # cinf_j(a_i, b_{h-1}) <= w_j
            constraints.add(names = ["c_cinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["cinf_%s_%s" % (aa.id, c.id),
                                   "w_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # cinf_j(a_i, b_{h-1}) <= dinf_{i,j}
            constraints.add(names = ["c_cinf2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["cinf_%s_%s" % (aa.id, c.id),
                                   "dinf_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # cinf_j(a_i, b_{h-1}) >= dinf_{i,j} - 1 + w_j
            constraints.add(names = ["c_cinf3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["cinf_%s_%s" % (aa.id, c.id),
                                   "dinf_%s_%s" % (aa.id, c.id),
                                   "w_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                           )

            # - M vinf_(i,j) < a_{i,j} - b_{h,j} - v_{h,j}
            constraints.add(names = ["v_vinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["vinf_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id),
                                   "v_%s_%s" % (b, c.id)],
                                  [-bigm, 1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [self.pt[aa.id].performances[c.id] -
                                   self.epsilon]
                           )

            # M (1 - vinf_(i,j)) >= a_{i,j} - b_{h,j} - v_{h,j}
            constraints.add(names = ["v_vinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["vinf_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id),
                                   "v_%s_%s" % (b, c.id)],
                                  [-bigm, 1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [self.pt[aa.id].performances[c.id] - bigm]
                           )

            # M dinf_(i,j) > a_{i,j} - b_{h-1,j}
            constraints.add(names = ["d_dinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["dinf_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id)],
                                  [bigm, 1]],
                                ],
                            senses = ["G"],
                            rhs = [self.pt[aa.id].performances[c.id] +
                                   self.epsilon]
                           )

            # M dinf_(i,j) <= a_{i,j} - b_{h-1,j} + M
            constraints.add(names = ["d_dinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["dinf_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id)],
                                  [bigm, 1]],
                                ],
                            senses = ["L"],
                            rhs = [self.pt[aa.id].performances[c.id] + bigm]
                           )

            # uinf_(a,j) <= vinf_(a,j)
            constraints.add(names = ["u_uinf_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["uinf_%s_%s" % (aa.id, c.id),
                                   "vinf_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # uinf_(a,j) <= z_j
            constraints.add(names = ["u_uinf2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["uinf_%s_%s" % (aa.id, c.id),
                                   "z_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # uinf_j(a_i, b_{h-1}) >= vinf_{i,j} - 1 + z_j
            constraints.add(names = ["u_uinf3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["uinf_%s_%s" % (aa.id, c.id),
                                   "vinf_%s_%s" % (aa.id, c.id),
                                   "z_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                           )
    def __add_alternative_upper_constraints(self, aa):
        constraints = self.lp.linear_constraints
        i = self.__categories.index(aa.category_id)
        b = self.__profiles[i]

        # sum csup_j(a_i, b_{h-1}) - osup_a < lambda + 2 (1 - alpha_i)
        constraints.add(names = ["gamma_sup_%s" % (aa.id)],
                        lin_expr =
                            [
                             [["csup_%s_%s" % (aa.id, c.id)
                               for c in self.criteria] + \
                              ["osup_%s" % (aa.id)] + \
                              ["lambda"] + ["a_%s" % aa.id],
                              [1 for c in self.criteria] + [-1] + [-1] + \
                              [2]],
                            ],
                        senses = ["L"],
                        rhs = [2 - self.epsilon]
                       )

        # sum usup(a,j) - LAMBDA >= M (osup_a - 1)
        constraints.add(names = ["veto_sup_%s" % (aa.id)],
                        lin_expr =
                            [
                             [["usup_%s_%s" % (aa.id, c.id)
                               for c in self.criteria] + \
                              ["LAMBDA"] + ["osup_%s" % (aa.id)],
                              [1 for c in self.criteria] + [-1] + [-2]],
                            ],
                        senses = ["G"],
                        rhs = [-2]
                       )

        # sum usup(a,j) - LAMBDA < M osup_a
        constraints.add(names = ["veto_sup2_%s" % (aa.id)],
                        lin_expr =
                            [
                             [["usup_%s_%s" % (aa.id, c.id)
                               for c in self.criteria] + \
                              ["LAMBDA"] + ["osup_%s" % (aa.id)],
                              [1 for c in self.criteria] + [-1] + [-2]],
                            ],
                        senses = ["L"],
                        rhs = [- self.epsilon]
                       )

        for c in self.criteria:
            bigm = self.ap_range.performances[c.id]

            # csup_j(a_i, b_h) <= w_j
            constraints.add(names = ["c_csup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["csup_%s_%s" % (aa.id, c.id),
                                   "w_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # csup_j(a_i, b_h) <= dsup_{i,j}
            constraints.add(names = ["c_csup2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["csup_%s_%s" % (aa.id, c.id),
                                   "dsup_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # csup_j(a_i, b_{h-1}) >= dsup_{i,j} - 1 + w_j
            constraints.add(names = ["c_csup3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["csup_%s_%s" % (aa.id, c.id),
                                   "dsup_%s_%s" % (aa.id, c.id),
                                   "w_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                           )

            # - M vsup_(i,j) < a_{i,j} - b_{h,j} - v_{h,j}
            constraints.add(names = ["v_vsup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["vsup_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id),
                                   "v_%s_%s" % (b, c.id)],
                                  [-bigm, 1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [self.pt[aa.id].performances[c.id] -
                                   self.epsilon]
                           )

            # M (1 - vsup_(i,j)) >= a_{i,j} - b_{h,j} - v_{h,j}
            constraints.add(names = ["v_vsup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["vsup_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id),
                                   "v_%s_%s" % (b, c.id)],
                                  [-bigm, 1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [self.pt[aa.id].performances[c.id] - bigm]
                           )

            # M dsup_(i,j) > a_{i,j} - b_{h,j}
            constraints.add(names = ["d_dsup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["dsup_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id)],
                                  [bigm, 1]],
                                ],
                            senses = ["G"],
                            rhs = [self.pt[aa.id].performances[c.id] +
                                   self.epsilon]
                           )

            # M dsup_(i,j) <= a_{i,j} - b_{h,j} + M
            constraints.add(names = ["d_dsup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["dsup_%s_%s" % (aa.id, c.id),
                                   "g_%s_%s" % (b, c.id)],
                                  [bigm, 1]],
                                ],
                            senses = ["L"],
                            rhs = [self.pt[aa.id].performances[c.id] + bigm]
                           )

            # usup_(a,j) <= vsup_(a,j)
            constraints.add(names = ["u_usup_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["usup_%s_%s" % (aa.id, c.id),
                                   "vsup_%s_%s" % (aa.id, c.id)],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # usup_(a,j) <= z_j
            constraints.add(names = ["u_usup2_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["usup_%s_%s" % (aa.id, c.id),
                                   "z_" + c.id],
                                  [1, -1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

            # usup_j(a_i, b_{h-1}) >= vsup_{i,j} - 1 + z_j
            constraints.add(names = ["u_usup3_%s_%s" % (aa.id, c.id)],
                            lin_expr =
                                [
                                 [["usup_%s_%s" % (aa.id, c.id),
                                   "vsup_%s_%s" % (aa.id, c.id),
                                   "z_" + c.id],
                                  [1, -1, -1]],
                                ],
                            senses = ["G"],
                            rhs = [-1]
                           )

    def add_alternatives_constraints(self):
        lower_cat = self.__categories[0]
        upper_cat = self.__categories[-1]

        for aa in self.aa:
            cat = aa.category_id

            if cat != lower_cat:
                self.__add_alternative_lower_constraints(aa)

            if cat != upper_cat:
                self.__add_alternative_upper_constraints(aa)

    def add_extra_constraints(self):
        constraints = self.lp.linear_constraints

        if self.model.veto_lbda is not None:
            constraints.add(names = ["LAMBDA"],
                            lin_expr =
                                [
                                 [["LAMBDA"],
                                  [1]]
                                 ],
                            senses = ["E"],
                            rhs = [self.model.veto_lbda]
                           )

        if self.model.veto_weights is not None:
            for c in self.criteria:
                constraints.add(names = ["z_%s" % c.id],
                                lin_expr =
                                    [
                                     [["z_%s" % c.id],
                                      [1]]
                                     ],
                                senses = ["E"],
                                rhs = [self.model.veto_weights[c.id].value]
                               )

        if self.indep_veto_weights is False:
            for c in self.criteria:
                constraints.add(names = ["z_%s" % c.id],
                                lin_expr =
                                    [
                                     [["w_%s" % c.id, "z_%s" % c.id],
                                      [1, -1]]
                                     ],
                                senses = ["E"],
                                rhs = [0]
                               )

    def add_constraints(self):
        constraints = self.lp.linear_constraints

        self.add_alternatives_constraints()

        profiles = self.cps.get_ordered_profiles()
        for h, c in product(range(len(profiles) - 1), self.criteria):
            # b_(h,j) <= b_(h+1,j)
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

            # b_(h,j) - v_(h,j) <= b_(h+1,j) - v_(h+1,j)
            constraints.add(names= ["dominance"],
                            lin_expr =
                                [
                                 [["g_%s_%s" % (profiles[h], c.id),
                                   "v_%s_%s" % (profiles[h], c.id),
                                   "g_%s_%s" % (profiles[h + 1], c.id),
                                   "v_%s_%s" % (profiles[h + 1], c.id)],
                                  [1, -1, -1, 1]],
                                ],
                            senses = ["L"],
                            rhs = [0]
                           )

        # sum w_j = 1
        constraints.add(names = ["wsum"],
                        lin_expr =
                            [
                             [["w_%s" % c.id for c in self.criteria],
                              [1 for c in self.criteria]],
                            ],
                        senses = ["E"],
                        rhs = [1]
                       )

        # sum z_j = 1
        constraints.add(names = ["wsum"],
                        lin_expr =
                            [
                             [["z_%s" % c.id for c in self.criteria],
                              [1 for c in self.criteria]],
                            ],
                        senses = ["E"],
                        rhs = [1]
                       )

    def add_objective(self):
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)

        self.lp.objective.set_linear([("a_%s" % aid, 1)
                                      for aid in self.aa.keys()])

        a1 = self.aa.get_alternatives_in_categories(self.__categories[1:])
        a2 = self.aa.get_alternatives_in_categories(self.__categories[:-1])

        if len(a1) > 0:
            self.lp.objective.set_linear([("oinf_%s" % a,
                                           - 1 / (2 * len(a1)))
                                         for a in a1])

        if len(a2) > 0:
            self.lp.objective.set_linear([("osup_%s" % a,
                                           - 1 / (2 * len(a2)))
                                         for a in a2])

    def solve(self):
        self.lp.solve()

        status = self.lp.solution.get_status()
        if status != self.lp.solution.status.MIP_optimal:
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

        pt = PerformanceTable()
        for p in self.__profiles:
            ap = AlternativePerformances(p)
            for c in self.criteria:
                perf = self.lp.solution.get_values("g_%s_%s" % (p, c.id))
                ap.performances[c.id] = round(perf, 5)
            pt.append(ap)

        self.model.bpt = pt
        self.model.bpt.update_direction(self.model.criteria)

        wv = CriteriaValues()
        for c in self.criteria:
            w = CriterionValue()
            w.id = c.id
            w.value = self.lp.solution.get_values('z_' + c.id)
            wv.append(w)

        self.model.veto_weights = wv
        self.model.veto_lbda = self.lp.solution.get_values("LAMBDA")

        v = PerformanceTable()
        for p in self.__profiles:
            vp = AlternativePerformances(p, {})
            for c in self.criteria:
                perf = self.lp.solution.get_values('v_%s_%s' % (p, c.id))
                vp.performances[c.id] = round(perf, 5)
            v.append(vp)

        self.model.veto = v

        return obj

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
    model2.cv = None
    model2.bpt = None
    model2.lbda = None
    model2.veto = None
    model2.veto_weights = None
    model2.veto_lbda = None

    mip = MipMRSortVC(model2, pt, aa, False)
    mip.solve()

    print model2.veto
    print model2.veto_lbda
    print model2.veto_weights
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
