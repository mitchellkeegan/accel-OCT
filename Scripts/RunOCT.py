"""Provides a command line interface for optimal tree models

This is intended to be run as a script from the command line with arguments setting the optimisation parameters

Includes an optional flag to run in validation mode in which case input validation is run on the arguments but the model
is not trained. This is for convenience to be able to construct sets of command line arguments from hyperparameters
and use the input validation to trim invalid arguments.
"""

import sys
import argparse

from datetime import datetime

from src.models.BendOCT import BendOCT
from src.models.BendRegOCT import BendRegOCT
from src.models.FlowRegOCT import FlowRegOCT
from src.utils.data import valid_datasets, load_instance
from src.utils.logging import save_optimisation_results, log_error

def package_gurobi_params(args):
    valid_input = True
    gurobi_params = {}

    if args.grb_Threads is not None:
        gurobi_params['Threads'] = args.grb_Threads

    if args.grb_NodeMethod is not None:
        gurobi_params['NodeMethod'] = args.grb_NodeMethod

    if args.grb_Method is not None:
        gurobi_params['Method'] = args.grb_Method

    if args.grb_MIPFocus is not None:
        gurobi_params['MIPFocus'] = args.grb_MIPFocus

    if args.grb_Seed is not None:
        if args.grb_Seed < 0:
            print(f'--grb_Seed value of {args.grb_Seed} is invalid. Please use a positive integer')
            valid_input = False
        else:
            gurobi_params['Seed'] = args.grb_Seed

    if args.grb_TimeLimit is not None:
        if args.grb_TimeLimit < 0:
            print(f'--grb_TimeLimit value of {args.grb_TimeLimit} is invalid. Please use a positive integer')
            valid_input = False
        else:
            gurobi_params['TimeLimit'] = args.grb_TimeLimit

    if args.grb_MIPGap is not None:
        if args.grb_MIPGap < 0:
            print(f'--grb_MIPGap value of {args.grb_MIPGap} is invalid. Please use a positive float')
            valid_input = False
        else:
            gurobi_params['MIPGap'] = args.grb_MIPGap

    if args.grb_Heuristics is not None:
        if args.grb_Heuristics < 0 or args.grb_Heuristics > 1:
            print(f'--grb_Heuristics value of {args.grb_Heuristics} is invalid. Please use a float in the range [0,1]')
            valid_input = False
        else:
            gurobi_params['Heuristics'] = args.grb_Heuristics

    gurobi_params['LogToFile'] = args.grb_LogToFile
    gurobi_params['LogToConsole'] = args.grb_LogToConsole

    return gurobi_params, valid_input

def package_opt_params(args):

    valid_input = True

    opt_params = {'Results Directory': args.ExperimentName,
                  'Use Baseline': args.UseBaseline,
                  'Debug Mode': args.Debug}

    if args.model in ['FlowRegOCT', 'BendRegOCT']:
        if args._lambda is None:
            print(f'A complexity penalty in [0,1] must be specified by -l or --lambda when using the {args.model} model')
            valid_input = False
        elif args._lambda < 0.0 or args._lambda > 1.0:
            print(f'--lambda value of {args._lambda} is invalid. Please choose complexity penalty in [0,1]')
            valid_input = False
        else:
            opt_params['lambda'] = args._lambda
    else:
        if args._lambda is not None:
            print(f'Cannot specify --lambda for the {args.model} model')
            valid_input = False

    if args.depth < 0:
        print(f'--depth value of {args.depth} is invalid. Please use a non-negative integer')
        valid_input = False
    else:
        opt_params['depth'] = args.depth

    if args.warmstart is not None:
        opt_params['Warmstart'] = True if args.warmstart == 'T' else False

    if args.PolishWarmstart is not None:
        if args.warmstart is not None and not args.warmstart:
            print('--PolishWarmstart=T not valid when -w=F')
            valid_input = False
        opt_params['Polish Warmstart'] = True if args.PolishWarmstart == 'T' else False

    if args.subjob_id is not None:
        opt_params['Subjob id'] = args.subjob_id

    if args.array_job_id is not None:
        opt_params['Array Job id'] = args.array_job_id

    return opt_params, valid_input

def package_callback_params(args, dataset_encoding_name):

    valid_input = True
    callback_settings = {}

    enhanced_benders_cuts_active = False

    if args.cb_Bend_EC is not None:
        callback_settings['Benders Cuts'] = {}
        callback_settings['Benders Cuts']['Enhanced Cuts'] = True if args.cb_Bend_EC == 'T' else False
        enhanced_benders_cuts_active = callback_settings['Benders Cuts']['Enhanced Cuts']

    if args.cb_Bend_ECL is not None:
        if enhanced_benders_cuts_active:
            callback_settings['Benders Cuts']['EC Level'] = args.cb_Bend_ECL
        else:
            print('Cannot set --cb_Bend_ECL when enhanced Benders cuts are not enabled by setting --cb_Bend_EC=T')
            valid_input = False

    if args.cb_Bend_Rw is not None:
        callback_settings['Benders Cuts']['Relax w'] = True if args.cb_Bend_Rw == 'T' else False

    cutset_callback_cuts_active = False

    if args.cb_CS is not None:
        callback_settings['Cut Set Callback'] = {}
        callback_settings['Cut Set Callback']['Enabled'] = True if args.cb_CS == 'T' else False
        cutset_callback_cuts_active = callback_settings['Cut Set Callback']['Enabled']

    if args.cb_CS_SM is not None:
        if cutset_callback_cuts_active:
            callback_settings['Cut Set Callback']['Solution Method'] = 'LP' if (args.cb_CS_SM == 'LP') else 'Dual Inspection'
        else:
            print('Cannot set --cb_CS_SM when cut-set callback cuts are not enabled by setting --cb_CS=T')
            valid_input = False

    if args.cb_CS_CT is not None:
        if cutset_callback_cuts_active:
            callback_settings['Cut Set Callback']['Cut Type'] = args.cb_CS_CT
        else:
            print('Cannot set --cb_CS_CT when cut-set callback cuts are not enabled by setting --cb_CS=T')
            valid_input = False

    if args.cb_Bend is not None:
        callback_settings['Benders Cuts'] = {}
        callback_settings['Benders Cuts']['Enabled'] = True if args.cb_Bend == 'T' else False

    solutionpolishing_callback_cuts_active = False

    if args.cb_SP is not None:
        callback_settings['Solution Polishing'] = {}
        callback_settings['Solution Polishing']['Enabled'] = True if args.cb_SP == 'T' else False
        solutionpolishing_callback_cuts_active = callback_settings['Solution Polishing']['Enabled']

        if callback_settings['Solution Polishing']['Enabled'] and args.depth <3:
            print(f'Solution Polishing D2S subroutine is not useful for trees with a depth of less than 3')
            valid_input = False

    if args.cb_SP_CV is not None:
        if solutionpolishing_callback_cuts_active:
            callback_settings['Solution Polishing']['Check Validity'] = True if (args.cb_SP_CV == 'T') else False
        else:
            print('Cannot set --cb_SP_CV when Solution Polishing is not enabled by setting --cb_SP=T')
            valid_input = False


    if args.cb_MNS is not None:
        callback_settings['Minimum Node Support'] = {}
        callback_settings['Minimum Node Support']['Enabled'] = True if args.cb_MNS == 'T' else False

    path_bound_cutting_planes_active = False

    if args.cb_PBCP is not None:
        callback_settings['Path Bound Cutting Planes'] = {}
        callback_settings['Path Bound Cutting Planes']['Enabled'] = True if args.cb_PBCP == 'T' else False

        path_bound_cutting_planes_active = callback_settings['Path Bound Cutting Planes']['Enabled']

    if args.cb_PBCP_BNS is not None:
        if args.cb_PBCP_BNS == 'T':
            if path_bound_cutting_planes_active:
                callback_settings['Path Bound Cutting Planes']['Bound Negative Samples'] = True
            else:
                print('Cannot set --cb_PBCP_BNS=T when Path Bound Cutting Planes are not enabled by setting --cb_PBCP=T')
                valid_input = False
        else:
            callback_settings['Path Bound Cutting Planes']['Bound Negative Samples'] = False


    if args.cb_PBCP_BSt is not None:
        if args.cb_PBCP_BSt == 'T':
            if path_bound_cutting_planes_active:
                callback_settings['Path Bound Cutting Planes']['Bound Structure'] = True
            else:
                print('Cannot set --cb_PBCP_BSt=T when Path Bound Cutting Planes are not enabled by setting --cb_PBCP=T')
                valid_input = False
        else:
            callback_settings['Path Bound Cutting Planes']['Bound Structure'] = False

    if args.cb_PBCP_EO is not None:
        if path_bound_cutting_planes_active:
            callback_settings['Path Bound Cutting Planes']['Endpoint Only'] = True if args.cb_PBCP_EO == 'T' else False
        else:
            print('Cannot set --cb_PBCP_EO when Path Bound Cutting Planes are not enabled by setting --cb_PBCP=T')
            valid_input = False

    if args.cb_PBCP_CT is not None:
        if path_bound_cutting_planes_active:
            callback_settings['Path Bound Cutting Planes']['Cut Type'] = args.cb_PBCP_CT
        else:
            print('Cannot set --cb_PBCP_CT when Path Bound Cutting Planes are not enabled by setting --cb_PBCP=T')
            valid_input = False

    if args.cb_SPLP is not None:
        callback_settings['Callback Subproblem LP'] = {}
        callback_settings['Callback Subproblem LP']['Enabled'] = True if args.cb_SPLP == 'T' else False

    if args.cb_SPDI is not None:
        callback_settings['Callback Subproblem Dual Inspection'] = {}
        callback_settings['Callback Subproblem Dual Inspection']['Enabled'] = True if args.cb_SPDI == 'T' else False

    return callback_settings, valid_input

def package_initial_cut_params(args, dataset_encoding_name):
    valid_input = True
    initial_cut_settings = {}

    cutset_initial_cuts_active = False

    if args.ic_CS is not None:
        initial_cut_settings['Cut Set Initial Cuts'] = {}
        initial_cut_settings['Cut Set Initial Cuts']['Enabled'] = True if args.ic_CS == 'T' else False
        cutset_initial_cuts_active = initial_cut_settings['Cut Set Initial Cuts']['Enabled']

    if args.ic_CS_SM is not None:
        if cutset_initial_cuts_active:
            initial_cut_settings['Cut Set Initial Cuts']['Solution Method'] = 'LP' if args.ic_CS_SM == 'LP' else 'Dual Inspection'
        else:
            print('Cannot set --ic_CS_SM when cut-set initial cuts are not enabled by setting --ic_CS=T')
            valid_input = False

    if args.ic_SPLP is not None:
        initial_cut_settings['Subproblem LP'] = {}
        initial_cut_settings['Subproblem LP']['Enabled'] = True if args.ic_SPLP == 'T' else False

    if args.ic_SPDI is not None:
        initial_cut_settings['Subproblem Dual Inspection'] = {}
        initial_cut_settings['Subproblem Dual Inspection']['Enabled'] = True if args.ic_SPDI == 'T' else False


    EQP_cuts_active = False
    if args.ic_EQP is not None:
        initial_cut_settings['EQP Initial Cuts'] = {}
        initial_cut_settings['EQP Initial Cuts']['Enabled'] = True if args.ic_EQP == 'T' else False
        EQP_cuts_active = initial_cut_settings['EQP Initial Cuts']['Enabled']

    if args.ic_EQP_HV is not None:
        if EQP_cuts_active:
            initial_cut_settings['EQP Initial Cuts']['H Variant'] = {'Ba': 'Basic',
                                                                     'Ch': 'Chain',
                                                                     'Re': 'Recursive'}[args.ic_EQP_HV]
        else:
            print('Cannot set --ic_EQP_HV when equivalent point initial cuts are not enabled by setting --ic_EQP=T')
            valid_input = False
    else:
        if EQP_cuts_active:
            print('Must set variant of H constraint set using --ic_EQP_HV when equivalent point initial cuts are enabled')
            valid_input = False

    if args.ic_EQP_FR is not None:
        if EQP_cuts_active:
            initial_cut_settings['EQP Initial Cuts']['Features Removed'] = args.ic_EQP_FR

            if (args.ic_EQP_FR == 0) and (args.ic_EQP_HV != 'Ba'):
                print('Must set --ic_EQP_HV=Ba when --ic_EQP_FR=0 in equivalent point constraints')
                valid_input = False
        else:
            print('Cannot set --ic_EQP_FR when equivalent point initial cuts are not enabled by setting --ic_EQP=T')
            valid_input = False
    else:
        if EQP_cuts_active:
            print('Must set number of features removed using --ic_EQP_FR when equivalent point initial cuts are enabled')
            valid_input = False

    if args.ic_EQP_DA is not None:
        if EQP_cuts_active:
            initial_cut_settings['EQP Initial Cuts']['Disaggregate Alpha'] = True if args.ic_EQP_DA == 'T' else False

            if (args.ic_EQP_DA == 'T') and (args.ic_EQP_HV == 'Ba'):
                print('Cannot set --ic_EQP_DA=T when --ic_EQP_HV=Ba in equivalent point constraints')
                valid_input = False
        else:
            print('Cannot set --ic_EQP_DA when equivalent point initial cuts are not enabled by setting --ic_EQP=T')
            valid_input = False
    else:
        if args.ic_EQP_HV in ['Ch', 'Re']:
            print('Must set --ic_EQP_DA when --ic_EQP_HV = "Re" or "Ch" in equivalent point initial cuts')
            valid_input = False

    if args.ic_EQP_GS is not None:
        if EQP_cuts_active:
            initial_cut_settings['EQP Initial Cuts']['Group Selection'] = True if args.ic_EQP_GS == 'T' else False
        else:
            print('Cannot set --ic_EQP_GS when equivalent point initial cuts are not enabled by setting --ic_EQP=T')
            valid_input = False

    if EQP_cuts_active:
        ignore_EQP_dataset_check = False

        if args.ic_EQP_IDC is not None:
            initial_cut_settings['EQP Initial Cuts']['Ignore Dataset Check'] = True if args.ic_EQP_IDC == 'T' else False
            ignore_EQP_dataset_check = initial_cut_settings['EQP Initial Cuts']['Ignore Dataset Check']

        # If --ic_EQP_IDC == 'T' then we do not confirm that the dataset actually has equivalent points for the given number of features removed.
        if not ignore_EQP_dataset_check:
            # Check that the instance is valid for the number of features removed
            EQP_bound_versions = [(EQP_cuts_active, args.ic_EQP_FR)]

            for cuts_active, features_removed in EQP_bound_versions:
                if cuts_active and features_removed is not None:
                    if dataset_encoding_name not in valid_datasets['eqp'][features_removed]:
                        print(f'Dataset {dataset_encoding_name} does not have any EQP sets for {features_removed} features removed')
                        valid_input = False


    return initial_cut_settings, valid_input

def package_args(args):

    all_inputs_valid = True

    # Encoded dataset name is empty so something can always be provided to initial cut argument packing function
    dataset_encoding_name = ''

    # Check that the dataset options are valid
    if args.dataset in valid_datasets['numerical'] + valid_datasets['mixed']:
        if args.encoding is None:
            all_inputs_valid = False
            print(f'Requested dataset {args.dataset} has continuous features and requires an encoding to be set with argument -e')
        if args.buckets is None:
            print((f'Requested dataset {args.dataset} has continuous features and requires '
                   f'the number of buckets in the encoding to be set with argument -b'))
            all_inputs_valid = False

        elif args.buckets < 2:
            print((f'Requested dataset {args.dataset} has continuous features and requires '
                   f'the number of buckets in the encoding to be at least 2'))
            all_inputs_valid = False

        # elif args.encoding is not None:
            # Dataset with continuous features has a valid encoding. Create the encoded instance string
        dataset_encoding_name = f'{args.dataset}_{args.encoding}_{args.buckets}'

    if args.dataset in valid_datasets['categorical']:
        if args.encoding is not None:
            print(f'Requested dataset {args.dataset} is categorical and cannot be used with encoding {args.encoding}')
            all_inputs_valid = False
        if args.buckets is not None:
            print(f'Requested dataset {args.dataset} is categorical and cannot be used with encoding bucket size {args.buckets}')
            all_inputs_valid = False

        # if args.encoding is None and args.buckets is None:
        # Categorical dataset has a valid encoding. Create the encoded instance string
        dataset_encoding_name = args.dataset

    callback, valid_input = package_callback_params(args, dataset_encoding_name)
    all_inputs_valid *= valid_input

    initial_cuts, valid_input = package_initial_cut_params(args, dataset_encoding_name)
    all_inputs_valid *= valid_input

    gurobi_params, valid_input = package_gurobi_params(args)
    all_inputs_valid *= valid_input

    opt_params, valid_input = package_opt_params(args)
    all_inputs_valid *= valid_input

    # Check if there were any initial cut or callback settings to be added
    if len(initial_cuts) > 0:
        opt_params['Initial Cuts'] = initial_cuts

    if len(callback) > 0:
        opt_params['Callback'] = callback

    return opt_params, gurobi_params, all_inputs_valid

def run_model(opt_model, dataset, encoding, buckets, opt_params, gurobi_params):

    print(opt_params)
    print(gurobi_params)

    if opt_model == 'BendOCT':
        model_class = BendOCT
    elif opt_model == 'FlowRegOCT':
        model_class = FlowRegOCT
    elif opt_model == 'BendRegOCT':
        model_class = BendRegOCT
    else:
        sys.exit(f'{opt_model} is not a valid model')

    Model = model_class(opt_params, gurobi_params)

    data = load_instance(dataset,
                         encoding_scheme=encoding,
                         num_buckets=buckets)

    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'RunOCT.py beginning model fit at {time_now}')

    fit_successful = Model.fit(data)

    if fit_successful is None:
        log_error(5)
    elif fit_successful:
        Model.post_process_model()
        save_optimisation_results(Model)
        Model.cleanup_GRB_environment()

def main(argv):

    parser = argparse.ArgumentParser(description='OCT command line interface. Many parameters are set to \'T\' or \'F\' as strings.\n'
                                                 'This is so that any optimisation parameters explicitly set by the use can be used'
                                                 'to generated the directory name for logs')

    opt_group = parser.add_argument_group('Optimisation Parameters')
    data_group = parser.add_argument_group('Dataset Parameters')
    gurobi_group = parser.add_argument_group('Gurobi Parameters',
                                             description='Set Gurobi parameters. grb_XX can be used to set Gurobi parameter XX')

    callback_benders_group = parser.add_argument_group('Callback Benders Cuts Parameters')
    callback_solution_polishing_group = parser.add_argument_group('Callback Solution Polishing Parameters')
    callback_PB_cutting_planes_group = parser.add_argument_group('Callback Path Bound Cutting Plane Parameters')
    callback_subproblemLP_group = parser.add_argument_group('Subproblem LP Callback Parameters')
    callback_subproblemDI_group = parser.add_argument_group('Subproblem Dual Inspection Callback Parameters')


    initial_cut_cutset_group = parser.add_argument_group('Cut-Set Inequality Initial Cut Parameters')
    callback_cutset_group = parser.add_argument_group('Cut-Set Inequality Callback Parameters')

    initial_cut_subproblemLP_group = parser.add_argument_group('Subproblem LP Initial Cut Parameters')
    initial_cut_subproblemDI_group = parser.add_argument_group('Subproblem Dual Inspection Initial Cut Parameters')

    initial_cut_EQP_group = parser.add_argument_group('Equivalent Point Initial Cut Parameters')

    # initial_cut_EQPB_group = parser.add_argument_group('Equivalent Point Basic Initial Cut Parameters')
    # initial_cut_EQPBG_group = parser.add_argument_group('Grouped Equivalent Point Basic Initial Cut Parameters')
    # initial_cut_EQPC_group = parser.add_argument_group('Equivalent Point Chain Initial Cut Parameters')

    # Model Choice
    parser.add_argument('model', type=str,
                        choices=['BendOCT', 'FlowRegOCT', 'BendRegOCT'])

    # Optimisation parameters
    opt_group.add_argument('ExperimentName', type=str)
    opt_group.add_argument('-d', '--depth', type=int, default=3,
                           help='Maximum tree depth')
    opt_group.add_argument('-l', '--lambda', type=float, dest='_lambda',
                           help='Per leaf penalty on accuracy')
    opt_group.add_argument('-w', '--warmstart',type=str,
                           choices=['T','F'],
                           help='Enable CART warm start solution')
    opt_group.add_argument('--PolishWarmstart', type=str,
                           choices=['T','F'],
                           help='Enable polishing of warmstart solution')
    opt_group.add_argument('--UseBaseline', action='store_true',
                           help='Override usefulness check of settings. Required for baseline experiments (No initial cuts or callback subroutines')
    opt_group.add_argument('--Debug', action='store_true',
                           help='Activate debug mode which writes decision variables to file')


    # Dataset parameters
    data_group.add_argument('dataset', type=str,
                           choices=valid_datasets['all'])
    data_group.add_argument('-e', '--encoding', type=str,
                            choices=['QT','QB'],
                            help='Encoding scheme for numerical datasets')
    data_group.add_argument('-b', '--buckets', type=int,
                            help='Number of buckets in encoding scheme')

    # Gurobi parameters
    gurobi_group.add_argument('--grb_TimeLimit', type=int)
    gurobi_group.add_argument('--grb_Threads', type=int,
                              choices=[i for i in range(0,1025)],
                              metavar = '[0-1024]')
    gurobi_group.add_argument('--grb_MIPGap', type=float)
    gurobi_group.add_argument('--grb_MIPFocus', type=int,
                              choices=[0,1,2,3])
    gurobi_group.add_argument('--grb_Heuristics', type=float)
    gurobi_group.add_argument('--grb_NodeMethod', type=int,
                              choices=[-1,0,1,2,3,4])
    gurobi_group.add_argument('--grb_Method', type=int,
                              choices=[-1,0,1,2,3,4])
    gurobi_group.add_argument('--grb_Seed', type=int)
    gurobi_group.add_argument('--grb_LogToConsole', action='store_true')
    gurobi_group.add_argument('--grb_LogToFile', action='store_true')

    # Callback Benders Cuts subroutine parameters
    callback_benders_group.add_argument('--cb_Bend', type=str,
                                        choices=['T', 'F'],
                                        help='When False, disable Benders cuts. WARNING - likely to break the model')
    callback_benders_group.add_argument('--cb_Bend_EC', type=str,
                                        choices=['T', 'F'],
                                        help='Enabled strengthened Benders cuts')
    callback_benders_group.add_argument('--cb_Bend_Rw', type=str,
                                        choices=['T', 'F'],
                                        help='Enabled/Disable relaxation of prediction variables')
    callback_benders_group.add_argument('--cb_Bend_ECL', type=int,
                                        choices=[1, 2],
                                        help='Level of strengthened Benders cuts')


    # Callback Solution Polishing subroutine parameters
    callback_solution_polishing_group.add_argument('--cb_SP', type=str,
                                                   choices=['T','F'])
    callback_solution_polishing_group.add_argument('--cb_SP_CV', type=str,
                                                   choices=['T', 'F'],
                                                   help='If True only run polish solutions which are valid w.r.t the full model (i.e. those which did not violate any Bender\'s cuts')


    parser.add_argument('--cb_MNS', type=str,
                        choices=['T', 'F'],
                        help='Enabled cutting off paths found in relaxations based on minimum sample requirements')

    # Callback Path Bound Cutting Planes subroutine parameters
    callback_PB_cutting_planes_group.add_argument('--cb_PBCP', type=str,
                                                   choices=['T', 'F'])
    callback_PB_cutting_planes_group.add_argument('--cb_PBCP_BNS', type=str,
                                                   choices=['T', 'F'],
                                                   help='Bound only incorrect samples instead of all samples in subtrees')
    callback_PB_cutting_planes_group.add_argument('--cb_PBCP_BSt', type=str,
                                                   choices=['T', 'F'],
                                                   help='Add cuts to enforce the structure of the subtree')
    callback_PB_cutting_planes_group.add_argument('--cb_PBCP_EO', type=str,
                                                  choices=['T', 'F'],
                                                  help='If True cutting planes are only added at the endpoints of integral paths')
    callback_PB_cutting_planes_group.add_argument('--cb_PBCP_CT', type=str,
                                                  choices=['Lazy', 'User'],
                                                  help='Callback cut type for cut-set inequalities. Must be one of "Lazy" (model.cbLazy) or "User" (model.cbCut)')

    # Callback cut-set inequality subroutine parameters
    callback_subproblemLP_group.add_argument('--cb_SPLP', type=str,
                                             choices=['T', 'F'],
                                             help='Generate cut-set inequalities by solving the subproblem LPs with Gurobi in root node')
    callback_subproblemDI_group.add_argument('--cb_SPDI', type=str,
                                             choices=['T', 'F'],
                                             help='Generate cut-set inequalities by solving the subproblem LPs by dual inspection in root node')

    initial_cut_cutset_group.add_argument('--ic_CS', type=str,
                                          choices=['T', 'F'],
                                          help='Generate cut-set inequalities by solving max-flow subproblems')
    initial_cut_cutset_group.add_argument('--ic_CS_SM', type=str,
                                          choices=['LP','DI'],
                                          help='Solution method for cut-set inequality subproblems. Must be one of "LP" (Solve LP using Gurobi) or "DI" (Solve min-cut problem by dual inspection)')

    callback_cutset_group.add_argument('--cb_CS', type=str,
                                          choices=['T', 'F'],
                                          help='Generate cut-set inequalities in callback by solving max-flow subproblems')
    callback_cutset_group.add_argument('--cb_CS_SM', type=str,
                                          choices=['LP', 'DI'],
                                          help='Solution method for cut-set inequality subproblems. Must be one of "LP" (Solve LP using Gurobi) or "DI" (Solve min-cut problem by dual inspection)')
    callback_cutset_group.add_argument('--cb_CS_CT', type=str,
                                       choices=['Lazy', 'User'],
                                       help='Callback cut type for cut-set inequalities. Must be one of "Lazy" (model.cbLazy) or "User" (model.cbCut)')

    # Cut-set inequality initial cuts parameters
    initial_cut_subproblemLP_group.add_argument('--ic_SPLP', type=str,
                                                choices=['T', 'F'],
                                                help='Generate cut-set inequalities by solving the subproblem LPs with Gurobi and add as initial cuts')
    initial_cut_subproblemDI_group.add_argument('--ic_SPDI', type=str,
                                                choices=['T', 'F'],
                                                help='Generate cut-set inequalities by solving the subproblem LPs by dual inspection and add as initial cuts')

    initial_cut_EQP_group.add_argument('--ic_EQP', type=str,
                                       choices=['T', 'F'],
                                       help='Enable equivalent point bounds')
    initial_cut_EQP_group.add_argument('--ic_EQP_FR', type=int,
                                       choices=[0, 1, 2],
                                       help='Maximum number of features allowed to be removed in equivalent point sets')
    initial_cut_EQP_group.add_argument('--ic_EQP_HV', type=str,
                                       choices=['Ba', 'Ch', 'Re'],
                                       help='Variant of constraint set H. One of Ba for basic, Ch for chain, or Re for recursive')
    initial_cut_EQP_group.add_argument('--ic_EQP_DA', type=str,
                                        choices=['T', 'F'],
                                        help='Enable disaggregation of alpha variables in Chain and Recursive variants of H constraints')
    initial_cut_EQP_group.add_argument('--ic_EQP_GS', type=str,
                                        choices=['T', 'F'],
                                        help='Enable group selection variant of G constraint set')
    initial_cut_EQP_group.add_argument('--ic_EQP_IDC', type=str,
                                       choices=['T','F'],
                                       help='Set to True (T) to run equivalent point initial cut generator regardless of what the dataset is (even if it doesn\'t have any equivalent points)')


    parser.add_argument('-j', '--subjob_id', type=int,
                        help='Job index when running from an array job')
    parser.add_argument('-v', '--validate', action='store_true',
                        help='Only run validation of arguments and do not call optimisation model')
    parser.add_argument('-a', '--array_job_id', type=str,
                        help='Identifier for job array ($PBS_ARRAY_ID)')
    parser.add_argument('--Dummy', action='store_true',
                        help='Identifies the subjob as a dummy job (i.e. not designed to actually be run) which allows for an array job to be run with only one actual subjob')


    args = parser.parse_args(argv)

    if args.Dummy:
        print('DUMMYJOB')
        return

    if not args.validate:
        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'RunOCT.py finished parsing arguments at {time_now}')

    opt_params, gurobi_params, inputs_valid = package_args(args)

    if args.validate:
        return inputs_valid

    if not inputs_valid:
        sys.exit('Exiting due to malformed input')

    if args.encoding == 'QB':
        encoding_scheme = 'Quantile Buckets'
    elif args.encoding == 'QT':
        encoding_scheme = 'Quantile Thresholds'
    else:
        encoding_scheme = None

    run_model(args.model, args.dataset, encoding_scheme, args.buckets, opt_params, gurobi_params)

if __name__ == '__main__':
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Entering RunOCT.py main() at {time_now}')
    main(sys.argv[1:])