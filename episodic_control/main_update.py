import run_experiment as expt
import argparse
import uuid

def main(experiment_type, env_type, rho, num_actions, arch, use_pvals, mem_temp, mem_envelope, RECORD=True):
    # experiment_parameters
    NUM_TRIALS = 10000
    NUM_EVENTS = 250

    # create run_id
    run_id = uuid.uuid4()
    # create experiment object
    if experiment_type == 0:
        ex = expt.training()
    else:
        ex = expt.testing(experiment_type)

    # run experiment
    ex.run(NUM_TRIALS, NUM_EVENTS)

    # log data
    if RECORD:
        expt.data_log(run_id, experiment_type, ex, load_from=load_id)

def process_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expt',
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        default=0,
                        type=int,
                        help="Experiment type")  # TODO describe expt types
    parser.add_argument('--env',
                        choices=['None', 'room'],
                        nargs='?',
                        default='None',
                        help='Environment type: None, room')
    parser.add_argument('--rho',
                        type=float,
                        nargs='?',
                        default=0.0,
                        help='Obstacle density value ')

    parser.add_argument('--num_actions',
                        type=int,
                        nargs='?',
                        default=4,
                        help='Number of available actions for agent')

    parser.add_argument('--arch',
                        type=str,
                        choices=['A', 'B'],
                        nargs='?',
                        default='B',
                        help='Agent architecture to use: A = no successor representation, B = with successor representation')

    parser.add_argument('--pvals',
                        type=bool,
                        nargs='?',
                        default=False,
                        help='Whether to decay memory relative to encoding time')

    parser.add_argument('--memtemp',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='Temperature value for EC softmax function')

    parser.add_argument('--memenv',
                        type=int,
                        nargs='?',
                        default=50,
                        help='Memory decay parameter (speed of forgetting)'
                        )

    args = parser.parse_args()

    return args


args = process_command_line_arguments()

if args.env == 'None':
    env_type = None
else:
    env_type = args.env

if __name__ == '__main__':
    main(args.expt, env_type, args.rho, args.num_actions, args.arch, args.pvals, args.memtemp, args.memenv, RECORD=False)
