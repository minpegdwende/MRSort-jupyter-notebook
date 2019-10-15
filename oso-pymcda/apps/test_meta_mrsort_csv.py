import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
import datetime
import random
import time
from collections import OrderedDict
from itertools import product
from pymcda.learning.meta_mrsort3 import MetaMRSortPop3, MetaMRSortPop3AUC
from pymcda.learning.meta_mrsortvc4 import MetaMRSortVCPop4
from pymcda.learning.heur_mrsort_init_profiles import HeurMRSortInitProfiles
from pymcda.learning.lp_mrsort_weights import LpMRSortWeights
from pymcda.learning.lp_mrsort_veto_weights import LpMRSortVetoWeights
from pymcda.learning.lp_mrsort_weights_auc import LpMRSortWeightsAUC
from pymcda.learning.heur_mrsort_profiles4 import MetaMRSortProfiles4
from pymcda.learning.heur_mrsort_profiles5 import MetaMRSortProfiles5
from pymcda.learning.heur_mrsort_veto_profiles5 import MetaMRSortVetoProfiles5
from pymcda.learning.lp_mrsort_mobius import LpMRSortMobius
from pymcda.learning.heur_mrsort_profiles_choquet import MetaMRSortProfilesChoquet
from pymcda.types import CriterionValue, CriteriaValues
from pymcda.types import Alternatives, Criteria, PerformanceTable
from pymcda.types import AlternativesAssignments, Categories
from pymcda.electre_tri import MRSort
from pymcda.generate import generate_alternatives
from pymcda.generate import generate_categories_profiles
from pymcda.generate import generate_random_profiles
from pymcda.generate import generate_random_criteria_weights
from pymcda.pt_sorted import SortedPerformanceTable
from pymcda.utils import compute_ca
from pymcda.utils import compute_ca_good
#from pymcda.utils import compute_confusion_matrix
from test_utils import test_result, test_results
from test_utils import load_mcda_input_data
from test_utils import save_to_xmcda

from pymcda.generate import generate_random_performance_table
from pymcda.generate import generate_random_criteria_weights
from pymcda.generate import generate_random_mrsort_model_with_coalition_veto
from pymcda.generate import generate_random_mrsort_model
from pymcda.generate import generate_random_mrsort_model_msjp
from pymcda.utils import compute_winning_and_loosing_coalitions
from pymcda.utils import compute_confusion_matrix, print_confusion_matrix
from pymcda.types import AlternativePerformances
import pdb
# from csv import reader, writer
from random_model_generation_msjp import create_random_model_mrsort
from random_model_generation_msjp import build_pymcda_instance_random
from random_model_generation_msjp import build_pymcda_instance_random_cipibm
from random_model_generation_msjp import build_pymcda_instance_random_duplicated



DATADIR = os.getenv('DATADIR', '%s/python_workspace/pymcda-master/pymcda-data' % os.path.expanduser('~'))

meta_mrsort = MetaMRSortVCPop4
#meta_mrsort = MetaMRSortPop3
#heur_init_profiles = HeurMRSortInitProfiles
#lp_weights = LpMRSortWeights
#heur_profiles = MetaMRSortProfiles5
#lp_veto_weights = LpMRSortVetoWeights
#heur_veto_profiles = MetaMRSortVetoProfiles5

aaa = dict()
allm = dict()

def run_test(seed, data, pclearning, nloop, nmodels, nmeta):
    random.seed(seed)
    global aaa
    global allm
    global fct_ca
    global LOO

    # Separate learning data and test data
    if LOO:
        pt_learning, pt_test = data.pt.split_LOO(seed)
    else:
        pt_learning, pt_test = data.pt.split(2, [pclearning, 100 - pclearning])
    aa_learning = data.aa.get_subset(pt_learning.keys())
    aa_test = data.aa.get_subset(pt_test.keys())

    #import pdb; pdb.set_trace()

    # Initialize a random model
    cat_profiles = generate_categories_profiles(data.cats)
    worst = data.pt.get_worst(data.c)
    best = data.pt.get_best(data.c)
    b = generate_alternatives(len(data.cats) - 1, 'b')
    bpt = None
    cvs = None
    lbda = None

    model = MRSort(data.c, cvs, bpt, lbda, cat_profiles)
    # if LOO:
    #     print(data.c, cvs, bpt, lbda, cat_profiles)
    #     print(model.categories_profiles.to_categories())
    #     print(model.categories)        
    #     import pdb; pdb.set_trace()

    # Run the metaheuristic
    t1 = time.time()

    pt_sorted = SortedPerformanceTable(pt_learning)

    # Algorithm
    meta = meta_mrsort(nmodels, model.criteria,
                       model.categories_profiles.to_categories(),
                       pt_sorted, aa_learning,
                       seed = seed * 100)
    # if LOO:
    #     print(nmodels, model.criteria,
    #                    model.categories_profiles.to_categories(),
    #                    pt_sorted, aa_learning)
        #import pdb; pdb.set_trace()
#lp_weights = lp_weights,
#heur_profiles = heur_profiles,
#lp_veto_weights = lp_veto_weights,
#heur_veto_profiles = heur_veto_profiles,

    for i in range(0, nloop):
        model, ca_learning, all_models = meta.optimize(nmeta, fct_ca)
        #import pdb; pdb.set_trace()

        if ca_learning == 1:
            break

    t_total = time.time() - t1

    aa_learning2 = model.pessimist(pt_learning)
    
    ca_learning = compute_ca(aa_learning, aa_learning2)
    ca_learning_good = compute_ca_good(aa_learning, aa_learning2)
    #import pdb; pdb.set_trace()
    auc_learning = model.auc(aa_learning, pt_learning)

    diff_learning = compute_confusion_matrix(aa_learning, aa_learning2,
                                           model.categories)

    # Compute CA of test setting
    
    if len(aa_test) > 0:
        aa_test2 = model.pessimist(pt_test)
        ca_test = compute_ca(aa_test, aa_test2)
        ca_test_good = compute_ca_good(aa_test, aa_test2)
        auc_test = model.auc(aa_test, pt_test)
        diff_test = compute_confusion_matrix(aa_test, aa_test2,
                                           model.categories)
        #import pdb; pdb.set_trace()

    else:
        ca_test = 0
        auc_test = 0
        ncat = len(data.cats)
        diff_test = OrderedDict([((a, b), 0) for a in model.categories \
                                             for b in model.categories])

    # Compute CA of whole set
    aa2 = model.pessimist(data.pt)
    ca = compute_ca(data.aa, aa2)
    ca_good = compute_ca_good(data.aa, aa2)
    auc = model.auc(data.aa, data.pt)
    diff_all = compute_confusion_matrix(data.aa, aa2, model.categories)

    t = test_result("%s-%d-%d-%d-%d-%d" % (data.name, seed, nloop, nmodels,
                                           nmeta, pclearning))

    model.id = 'learned'
    aa_learning.id, aa_test.id = 'learning_set', 'test_set'
    pt_learning.id, pt_test.id = 'learning_set', 'test_set'
    save_to_xmcda("%s/%s.bz2" % (directory, t.test_name),
                  model, aa_learning, aa_test, pt_learning, pt_test)

    t['seed'] = seed
    t['na'] = len(data.a)
    t['nc'] = len(data.c)
    t['ncat'] = len(data.cats)
    t['pclearning'] = pclearning
    t['nloop'] = nloop
    t['nmodels'] = nmodels
    t['nmeta'] = nmeta
    t['na_learning'] = len(aa_learning)
    t['na_test'] = len(aa_test)
    t['ca_learning'] = ca_learning
    t['ca_test'] = ca_test
    t['ca_all'] = ca    
    t['ca_learning_good'] = ca_learning_good
    t['ca_test_good'] = ca_test_good
    t['ca_all_good'] = ca_good
    t['auc_learning'] = auc_learning
    t['auc_test'] = auc_test
    t['auc_all'] = auc

    # import pdb; pdb.set_trace()
    aaa[seed]=dict()
    aaa[seed]['id'] = seed
    aaa[seed]['learning_asgmt_id'] = [i.id for i in aa_learning]
    aaa[seed]['learning_asgmt'] = [i.category_id for i in aa_learning]
    aaa[seed]['learning_asgmt2'] = [i.category_id for i in aa_learning2]        
    aaa[seed]['test_asgmt_id'] = [i.id for i in aa_test]
    aaa[seed]['test_asgmt'] = [i.category_id for i in aa_test]
    aaa[seed]['test_asgmt2'] = [i.category_id for i in aa_test2]
    aaa[seed]['criteria'] =  [i for i,j in model.criteria.items()]
    aaa[seed]['criteria_weights'] = [str(i.value) for i in model.cv.values()]
    aaa[seed]['profiles_values'] = [str(model.bpt['b1'].performances[i]) for i,j in model.criteria.items()]
    aaa[seed]['lambda'] = model.lbda
    #[model.bpt['b1'].performances[i] for i,j in model.criteria.items()]


    allm[seed]=dict()
    allm[seed]['id'] = seed
    current_model = 0
    allm[seed]['mresults'] = dict()
    for all_model in list(all_models)[1:]:
        current_model += 1 # skipping the 1rst model already treated
        allm[seed]['mresults'][current_model] = ["",""]
        aa_learning2_allm = all_model.model.pessimist(pt_learning)
        ca_learning_allm = compute_ca(aa_learning, aa_learning2_allm)
        ca_learning_good_allm = compute_ca_good(aa_learning, aa_learning2_allm)
        auc_learning_allm = all_model.model.auc(aa_learning, pt_learning)
        # diff_learning_allm = compute_confusion_matrix(aa_learning, aa_learning2_allm,
        #                                        all_model.model.categories)
        # Compute CA of test setting
        if len(aa_test) > 0:
            aa_test2_allm = all_model.model.pessimist(pt_test)
            ca_test_allm = compute_ca(aa_test, aa_test2_allm)
            ca_test_good_allm = compute_ca_good(aa_test, aa_test2_allm)
            auc_test_allm = all_model.model.auc(aa_test, pt_test)
            # diff_test_allm = compute_confusion_matrix(aa_test, aa_test2_allm,
            #                                    all_model.categories)
        else:
            ca_test_allm = 0
            auc_test_allm = 0
            ncat_allm = len(data.cats)
            # diff_test_allm = OrderedDict([((a, b), 0) for a in all_model.categories \
            #                                      for b in all_model.model.categories])
        # Compute CA of whole set
        aa2_allm = all_model.model.pessimist(data.pt)
        ca_allm = compute_ca(data.aa, aa2_allm)
        ca_good_allm = compute_ca_good(data.aa, aa2_allm)
        auc_allm = all_model.model.auc(data.aa, data.pt)
        #diff_all_allm = compute_confusion_matrix(data.aa, aa2_allm, all_model.model.categories) 
        allm[seed]['mresults'][current_model][0] = 'na_learning,na_test,ca_learning,ca_test,ca_all,ca_learning_good,ca_test_good,ca_all_good,auc_learning,auc_test,auc_all'
        allm[seed]['mresults'][current_model][1] =  str(len(aa_learning)) + "," + str(len(aa_test)) + "," + str(ca_learning_allm) +  "," + str(ca_test_allm) +  "," + str(ca_allm) + "," + str(ca_learning_good_allm) + "," + str(ca_test_good_allm) + "," + str(ca_good_allm) +  "," + str(auc_learning_allm) + "," + str(auc_test_allm) +  "," + str(auc_allm)
        #allm[seed]['mresults'][current_model][1] =
        #all_model.model.bpt['b1'].performances
        #all_model.model.cv.values()
        #import pdb; pdb.set_trace()

        # allm[seed][current_model]['na_learning'] = len(aa_learning)
        # allm[seed][current_model]['na_test'] = len(na_test)
        # allm[seed][current_model]['ca_learning'] = ca_learning_allm
        # allm[seed][current_model]['ca_test'] = ca_test_allm
        # allm[seed][current_model]['ca_all'] = ca_allm     
        # allm[seed][current_model]['ca_learning_good'] = ca_learning_good_allm
        # allm[seed][current_model]['ca_test_good'] = ca_test_good_allm
        # allm[seed][current_model]['ca_all_good'] = ca_good_allm
        # allm[seed][current_model]['auc_learning'] = auc_learning_allm
        # allm[seed][current_model]['auc_test'] = auc_test_allm
        # allm[seed][current_model]['auc_all'] = auc_allm


    for k, v in diff_learning.items():
        t['learn_%s_%s' % (k[0], k[1])] = v
    for k, v in diff_test.items():
        t['test_%s_%s' % (k[0], k[1])] = v
    for k, v in diff_all.items():
        t['all_%s_%s' % (k[0], k[1])] = v

    t['t_total'] = t_total

    return t

def run_tests(nseeds, data, pclearning, nloop, nmodels, nmeta, filename):
    # Create the CSV writer
    f = open(filename, 'wb')
    writer = csv.writer(f)
    global aaa

    # Write the test options
    writer.writerow(['data', data.name])
    writer.writerow(['meta_mrsort', meta_mrsort.__name__])
    writer.writerow(['nloop', nloop])
    writer.writerow(['nmodels', nmodels])
    writer.writerow(['nmeta', nmeta])
    writer.writerow(['pclearning', pclearning])

    # Create a test_results instance
    results = test_results()

    # Initialize the seeds
    seeds = range(nseeds)

    # Run the algorithm
    initialized = False
    for _pclearning, _nloop, _nmodels, _nmeta, seed in product(pclearning,
                                        nloop, nmodels, nmeta, seeds):
        t1 = time.time()
        t = run_test(seed, data, _pclearning, _nloop, _nmodels, _nmeta)
        t2 = time.time()

        if initialized is False:
            fields = t.get_attributes()
            writer.writerow(fields)
            initialized = True

        t.tocsv(writer, fields)
        f.flush()
        print("%s (%5f seconds)" % (t, t2 - t1))

        results.append(t)

    # Perform a summary
    writer.writerow(['' , ''])

    t = results.summary(['na', 'nc', 'ncat', 'pclearning', 'na_learning',
                         'na_test', 'nloop', 'nmodels', 'nmeta'],
                        ['ca_learning', 'ca_learning_good', 'ca_test', 'ca_test_good' , 'ca_all', 'ca_all_good',
                         'auc_learning', 'auc_test', 'auc_all', 't_total'])
    writer.writerow(['', ''])

    t.tocsv(writer)

if __name__ == "__main__":
    from optparse import OptionParser
    from test_utils import read_single_integer, read_multiple_integer
    from test_utils import read_csv_filename

    parser = OptionParser(usage = "python %s [options]" % sys.argv[0])
    parser.add_option("-c", "--choquet", action = "store_true",
                      dest = "choquet", default = False,
                      help = "use MR-Sort Choquet")
    parser.add_option("-i", "--csvfile", action = "store", type="string",
                      dest = "csvfile",
                      help = "csv file with data")
    parser.add_option("-p", "--pclearning", action = "store", type="string",
                      dest = "pclearning",
                      help = "Percentage of data to use in learning set")
    parser.add_option("-m", "--nmodels", action = "store", type="string",
                      dest = "nmodels",
                      help = "Size of the population (of models)")
    parser.add_option("-l", "--max-loops", action = "store", type="string",
                      dest = "max_loops",
                      help = "Max number of loops of the whole "
                             "metaheuristic")
    parser.add_option("-o", "--max_oloops", action = "store", type="string",
                      dest = "max_oloops",
                      help = "max number of loops for the metaheuristic " \
                             "used to find the profiles")
    parser.add_option("-s", "--nseeds", action = "store", type="string",
                      dest = "nseeds",
                      help = "number of seeds")
    parser.add_option("-f", "--filename", action = "store", type="string",
                      dest = "filename",
                      help = "filename to save csv output")
    (options, args) = parser.parse_args()



    ################################ CODE ADDED RAND
    ################################ CODE ADDED RAND

    # cipibm_dir = "/rand_na100_ncat2_ncr0-10/"

    #model,pt,cipibm_dir = create_random_model_mrsort(100, 2, 10, random_directions = [-1]*0+[1]*10)
    model,pt,cipibm_dir = create_random_model_mrsort(100, 2, 10, random_directions = [1]*10+[-1]*0)
    #options.csvfile = build_pymcda_instance_random(model,pt, cipibm_dir = cipibm_dir)
    options.csvfile = build_pymcda_instance_random_duplicated(model,pt, cipibm_dir = cipibm_dir)
    # import pdb; pdb.set_trace()
    ################################ CODE ADDED RAND
    ################################ CODE ADDED RAND


    while not options.csvfile:
        options.csvfile = raw_input("CSV file containing data ? ")
    
    data = load_mcda_input_data(options.csvfile)
    if data is None:
        exit(1)

    if options.choquet is True:
        lp_weights = LpMRSortMobius
        heur_profiles = MetaMRSortProfilesChoquet

    # options.pclearning = read_multiple_integer(options.pclearning,
    #                                            "Percentage of data to " \
    #                                            "use in the learning set")
    # options.max_oloops = read_multiple_integer(options.max_oloops, "Max " \
    #                                            "number of loops for " \
    #                                            "profiles' metaheuristic")
    # options.nmodels = read_multiple_integer(options.nmodels,
    #                                         "Population size (models)")
    # options.max_loops = read_multiple_integer(options.max_loops, "Max " \
    #                                           "number of loops for the " \
    #                                           "whole metaheuristic")
    # options.nseeds = read_single_integer(options.nseeds, "Number of seeds")

    #for p,m,l,ll in products(range(),range(),range(), range(), ):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    #os.mkdir(DATADIR + "/uniq_inst_81_27/")
    #import pdb; pdb.set_trace()

    # Loss function : defines how to optimize the CA
    fct_ca = 0 # sort_models
    # fct_ca = 1 # sort_models_ca0_1
    # fct_ca = 2 # sort_models_ca1_1000
    # fct_ca = 3 # sort_models_ca1_1

    #choosing to run LOO (Leave-One-Out)
    LOO = False

    #cipibm_dir = "/industry/"
    options.max_loops = [10]
    options.max_oloops = [5]
    options.nmodels = [10]
    options.nseeds = 10
    options.pclearning = [20,50,80,0]

    for it in range(4):
        if it == 3:
            LOO = True
            options.nmodels = [10]
            options.max_loops = [10]
            options.max_oloops = [5]

        cipibm_dir2 = DATADIR + cipibm_dir + "p" + str(options.pclearning[it]) + "_" + "m" + str(options.nmodels[0]) + "_" + "l" + str(options.max_loops[0]) + "_" + "ll" + str(options.max_oloops[0]) + "_" + "s" + str(options.nseeds)
        if not os.path.exists(cipibm_dir2):
            os.mkdir(cipibm_dir2)

        new_DATADIR = cipibm_dir2
        #new_DATADIR = DATADIR

        default_filename = "%s/test_meta_mrsort3-%s-%s.csv" \
                           % (new_DATADIR, data.name, dt)
        alt_filename = "%s/alt_test_meta_mrsort3-%s-%s.csv" \
                           % (new_DATADIR, data.name, dt)
        allm_filename = "%s/allm_test_meta_mrsort3-%s-%s.csv" \
                           % (new_DATADIR, data.name, dt)
        options.filename = read_csv_filename(options.filename, default_filename)

        directory = options.filename + "-data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if LOO:
            run_tests(len(data.a), data, [options.pclearning[it]], options.max_loops,
                  options.nmodels, options.max_oloops, default_filename)
        else :
            run_tests(options.nseeds, data, [options.pclearning[it]], options.max_loops,
                  options.nmodels, options.max_oloops, default_filename)





        nb_right_crit_dir = [0]*(len(aaa.values()))
        w_right_crit_dir = [0]*(len(aaa.values()))
        nb_null_weights = [0]*(len(aaa.values()))
        f = open(alt_filename, 'w')
        writer = csv.writer(f, delimiter = " ")
        writer.writerow(['na', len(data.a)])
        writer.writerow(['nc', len(data.c)])
        writer.writerow(['ncat', len(data.cats)])
        writer.writerow(['nloop', options.max_loops[0]])
        writer.writerow(['nmodels', options.nmodels[0]])
        writer.writerow(['nmeta', options.max_oloops[0]])
        writer.writerow(['pclearning', options.pclearning[it]])        seed_elm = 0
        for sss in aaa.values():
            #import pdb; pdb.set_trace()
            writer.writerow(['current_seed,', sss['id']])
            writer.writerow(['learning_asgmt_id,', ",".join(sss['learning_asgmt_id'])])
            writer.writerow(['learning_asgmt,', ",".join(sss['learning_asgmt'])])
            writer.writerow(['learning_asgmt2,', ",".join(sss['learning_asgmt2'])])
            writer.writerow(['test_asgmt_id,', ",".join(sss['test_asgmt_id'])])
            writer.writerow(['test_asgmt,', ",".join(sss['test_asgmt'])])
            writer.writerow(['test_asgmt2,', ",".join(sss['test_asgmt2'])])
            writer.writerow(['criteria,', ",".join(sss['criteria'])])
            # to be more generallist, need to make a loop on the b(i)
            writer.writerow(['original_profiles_values,', ",".join([str(model.bpt['b1'].performances[i]) for i,j in model.criteria.items()])])
            #print([str(model.bpt['b1'].performances) for i,j in model.criteria.items()])
            writer.writerow(['learned_profiles_values,', ",".join(sss['profiles_values'])])
            #print(sss['profiles_values'])
            writer.writerow(['original_criteria_weights,', ",".join([str(i.value) for i in model.cv.values()])])
            #print([str(model.bpt['b1'].performances) for i,j in model.criteria.items()])])])
            #print([str(i) for i in model.cv.values()])
            writer.writerow(['learned_criteria_weights,', ",".join(sss['criteria_weights'])])
            #tmp_perc = 0
            #writer.writerow(['original_criteria_weights,', ",".join([str(i.value) for i in model.cv.values()])])
            #tmp_ori_weights = [str(i.value) for i in model.cv.values()] + [0]*(len(sss['criteria_weights']))
            #model.criteria.values()

            for i in range(len(model.criteria)):
                # check if the current criterion has a duplucate ??
                # if yes, give details on the duplicated criterion  
                c_dupl = i + len(model.criteria)
                if model.criteria.values()[i].direction == 1: # avoir le model en entier ici pour verifier directement entre deux direction
                    if float(sss['criteria_weights'][i]) != 0 and float(sss['criteria_weights'][c_dupl]) == 0:
                        nb_right_crit_dir[seed_elm] += 1
                        w_right_crit_dir[seed_elm] += float(sss['criteria_weights'][i])
                if model.criteria.values()[i].direction == -1: # avoir le model en entier ici pour verifier directement entre deux direction
                    if float(sss['criteria_weights'][i]) == 0 and float(sss['criteria_weights'][c_dupl]) != 0:
                        nb_right_crit_dir[seed_elm] += 1
                        w_right_crit_dir[seed_elm] += float(sss['criteria_weights'][i])
                if float(sss['criteria_weights'][i]) == 0 and float(sss['criteria_weights'][c_dupl]) == 0: #if criteria are duplicated otherwise just one check is necessary
                    nb_null_weights[seed_elm] += 1 


            writer.writerow(['original_lambda,', model.lbda])
            writer.writerow(['learned_lambda,', sss['lambda']])
            writer.writerow(['%_right_crit_dir,', str((nb_right_crit_dir[seed_elm]*1.0)/len(model.criteria))])
            writer.writerow(['w_right_crit_dir,', str(w_right_crit_dir[seed_elm])])
            writer.writerow(['%_null_weights,', str((nb_null_weights[seed_elm]*1.0)/len(model.criteria))])
            #import pdb; pdb.set_trace()
            #print(sss['criteria_weights'])
            seed_elm += 1

        writer.writerow(['%_right_crit_dir_avg,', str(sum(nb_right_crit_dir)/len(nb_right_crit_dir))])
        writer.writerow(['w_right_crit_dir_avg,', str(sum(w_right_crit_dir)/len(w_right_crit_dir))])
        writer.writerow(['%_null_weights_avg,', str(sum(nb_null_weights)/len(nb_null_weights))])

        f = open(allm_filename, 'w')
        writer = csv.writer(f, delimiter = " ")
        writer.writerow(['na', len(data.a)])
        writer.writerow(['nc', len(data.c)])
        writer.writerow(['ncat', len(data.cats)])
        writer.writerow(['nloop', options.max_loops[0]])
        writer.writerow(['nmodels', options.nmodels[0]])
        writer.writerow(['nmeta', options.max_oloops[0]])
        writer.writerow(['pclearning', options.pclearning[it]])

        for mmm in allm.values():
            #import pdb; pdb.set_trace()
            writer.writerow(['current_seed,', mmm['id']])
            if mmm['mresults'] and mmm['mresults'][1]:
                writer.writerow([',,', mmm['mresults'][1][0]])
            for kmmm in mmm['mresults']:
                writer.writerow(['id_model,', str(kmmm) + ',', mmm['mresults'][kmmm][1]])

        print("Results saved in '%s'" % options.filename)

