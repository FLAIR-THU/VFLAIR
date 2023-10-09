import re
import numpy as np
import statistics
import glob

for path in glob.glob("tmpoutput/*"):
    print(path)
    with open(path, mode="r") as f:
        data = f.read()

    # Initialize lists to store data
    training_times = []
    test_accs = []
    node_counts = []
    num_comms = []
    
    # Regular expressions to match relevant lines
    time_pattern = re.compile(r'training time: ([\d.]+) \[s\]')
    test_acc_pattern = re.compile(r'test acc: ([\d.]+)')
    node_pattern = re.compile(r'#Nodes: (\d+)')
    num_comm = re.compile(r'Comm: \((\d+),')

    # Split the data into individual records
    records = data.split('\ntype of model: ')

    # Loop through each record and extract relevant data
    for record in records:
        time_match = time_pattern.search(record)
        test_acc_match = test_acc_pattern.search(record)
        node_match = node_pattern.search(record)
        numcom_match = num_comm.search(record)

        if time_match and test_acc_match and node_match:
            training_times.append(float(time_match.group(1)))
            test_accs.append(float(test_acc_match.group(1)))
            node_counts.append(int(node_match.group(1)))
            num_comms.append(int(numcom_match.group(1)))
    # Calculate the average and standard deviation
    avg_training_time = statistics.mean(training_times)
    try:
        std_dev_training_time = statistics.stdev(training_times)
    except:
        std_dev_training_time = 0.0

    avg_test_acc = statistics.mean(test_accs)
    try:
        std_dev_test_acc = statistics.stdev(test_accs)
    except:
        std_dev_test_acc = 0.0

    avg_node_count = statistics.mean(node_counts)
    try:
        std_dev_node_count = statistics.stdev(node_counts)
    except:
        std_dev_node_count = 0

    avg_numcom = statistics.mean(num_comms)
    std_dev_numcom = statistics.stdev(num_comms)

    print(avg_training_time, avg_test_acc, avg_node_count, avg_numcom)
    print(std_dev_training_time, std_dev_test_acc, std_dev_node_count, std_dev_numcom)
