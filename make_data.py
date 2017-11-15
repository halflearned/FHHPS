from data_utils import clock_seed, generate_data
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
                      
    parser.add_argument('-n', '--num_obs',
                        help='Number of observations',
                        required=False,
                        default=10000,
                        type=int)
    
    parser.add_argument('-out','--output_filename', 
                        help='Output file name',
                        required=False,
                        default='fake_data',
                        type=str)

    parser.add_argument('-s','--seed', 
                        help='Random seed',
                        required=False,
                        default=None,
                        type=int)

    parser.add_argument('-EU', '--EU',
                        help='Average of second period shock to intercept.',
                        required=False,
                        default=.3,
                        type=float)
    parser.add_argument('-EV', '--EV',
                        help='Average of second period shock to intercept.',
                        required=False,
                        default=.1,
                        type=float)
    parser.add_argument('-EA', '--EA',
                        help='Average of first-period intercept.',
                        required=False,
                        default=2,
                        type=float)
    parser.add_argument('-EB', '--EB',
                        help='Average of first-period slope.',
                        required=False,
                        default=.4,
                        type=float)
    parser.add_argument('-EX1', '--EX1',
                        help='Average of first-period regressor.',
                        required=False,
                        default=0,
                        type=float)
    parser.add_argument('-EX2', '--EX2',
                        help='Average of second-period regressor.',
                        required=False,
                        default=0,
                        type=float)
    
    parser.add_argument('-VA', '--VA',
                        help='Variance of first-period intercept.',
                        required=False,
                        default=9,
                        type=float)
    parser.add_argument('-VB', '--VB',
                        help='Variance of first-period slope.',
                        required=False,
                        default=.4,
                        type=float)
    parser.add_argument('-VU', '--VU',
                        help='Variance of second-period shocks to intercept.',
                        required=False,
                        default=1,
                        type=float)
    parser.add_argument('-VV', '--VV',
                        help='Variance of second-period shocks to slope.',
                        required=False,
                        default=.1,
                        type=float)
    
    parser.add_argument('-rho', '--rho',
                        help='Correlation between all variables except between second-period shocks and everything else.',
                        required=False,
                        default=.5,
                        type=float)
    

    
    args = parser.parse_args()
    
    if args.seed is None:
        seed = clock_seed()
    else:
        seed = args.seed

    df = generate_data(n = args.num_obs, 
                         EA = args.EA, 
                         EB = args.EB,
                         EX1 = args.EX1,
                         EX2 = args.EX2,
                         EU = args.EU,
                         EV = args.EV, 
                         vA = args.VA,
                         vB = args.VB,
                         vU = args.VU,
                         vV = args.VV,
                         rho = args.rho,
                         seed = seed)
    
    df.to_csv(args.output_filename + ".csv")
    
    print("Created data file '{}.csv'. \nIts descriptive statistics are:".format(args.output_filename))
    print(df.describe())
    
