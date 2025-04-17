import configparser

def input_parser(input_file):
    '''
    Parse the input file and returns the parameters
    -----------------------------------------------
    Parameters: 
    input_file: string, name of the input file
    ------------------------------------------
    Returns:
    A dictionary of the Neural network parameters, input and output file path and name.
    '''
    # add logger.info
    My_config = configparser.ConfigParser()
    My_config.read(input_file)
    parser = dict()

    training_info = My_config['TRAINING_INFO']
    parser['train_path'] = training_info.get('train_path')
    parser['train_data_path'] = training_info.get('train_data_path')
    parser['train_data_file'] = training_info.get('train_data_file')
    parser['training_path'] = training_info.get('training_path')
    parser['save_frequency'] = training_info.get('save_frequency')

     
    # descriptor_info = My_config['DESCRIPTOR_INFO'] ... later
    
    network_info = My_config['NETWORK_PARAMETERS']
    parser['n_layers'] = network_info.getint('n_layers')
    nodes = network_info.get('n_nodes')
    parser['n_nodes'] = [int(n) for n in nodes.split(',')]
    parser['restart'] = network_info.getboolean('restart', False)
    parser['batch_size'] =  network_info.getint('batch_size', 8)
    parser['learning_rate'] = network_info.getfloat('learning_rate', 0.001)
    parser['decay_after'] = network_info.getint('decay_after', 10000)
    parser['n_epochs'] = network_info.getint('n_epochs', 100)

    return parser


