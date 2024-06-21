import csv
import os

def create_row_head(seq_len, pred_len):
    row_head = []
    contents = ['index', 'unit1_rgb', 'unit1_radar', 'unit1_lidar', 'unit1_loc', 'unit2_loc', 'unit1_pwr_60ghz', 'unit1_beam']
    for c in contents:
        if c in ['index', 'unit1_loc', 'unit1_beam']:
            row_head.append(c)
        elif c in ['unit1_rgb', 'unit1_radar', 'unit1_lidar']:
            for i in range(1, seq_len+1):
                row_head.append(c+'_'+str(i))
        elif c in ['unit1_pwr_60ghz']:
            for j in range(1, pred_len+1):
                row_head.append(c+'_'+str(j))
        elif c in ['unit2_loc']:
            for i in range(1, 3):
                row_head.append(c+'_'+str(i))
    # print(row_head)
    return row_head

def list2dict(datalist):
    id_data_dict = {}
    for data_name in datalist:
        if data_name.endswith(('.jpg', '.npy', '.ply', '.txt')):
            id = int(data_name.split('.')[0].split('_')[-1])
            id_data_dict[id] = data_name
    return id_data_dict

def get_beam_label(beam_data, root):
    beam_label = []
    for data_path in beam_data:
        with open(os.path.join(root, data_path), 'r') as f:
            data = f.readlines()
        beam_max = max(data)
        beam_max_id = data.index(beam_max)
        beam_label.append(str(beam_max_id+1))
    beam_label = '_'.join(beam_label)
    return [beam_label]

def createRootCSV(RootPath, OutputFile, seq_len, pred_len):
    SaveOutputFile = os.path.join(RootPath, OutputFile)
    # write the row head into csv
    row_head = create_row_head(seq_len, pred_len)
    with open(SaveOutputFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_head)

    # different set has different scenarios
    if 'dev_multi_modal' in OutputFile:
        scenario_list = ['scenario32', 'scenario33', 'scenario34']
    elif 'data_adaptation_multi_modal' in OutputFile:
        scenario_list = ['scenario31', 'scenario32', 'scenario33']

    index = 1
    for scen in scenario_list:
        # find all data file
        camera_data_list = os.listdir(os.path.join(RootPath, scen, 'unit1', 'camera_data'))
        radar_data_list = os.listdir(os.path.join(RootPath, scen, 'unit1', 'radar_data'))
        lidar_data_list = os.listdir(os.path.join(RootPath, scen, 'unit1', 'lidar_data'))
        gps_u2_data_list = os.listdir(os.path.join(RootPath, scen, 'unit2', 'GPS_data'))
        beam_data_list = os.listdir(os.path.join(RootPath, scen, 'unit1', 'mmWave_data'))

        # convert list to dict for ID convenience
        camera_id_data_dict = list2dict(camera_data_list)
        radar_id_data_dict = list2dict(radar_data_list)
        lidar_id_data_dict = list2dict(lidar_data_list)
        gps_u2_id_data_dict = list2dict(gps_u2_data_list)
        beam_id_data_dict = list2dict(beam_data_list)
        beam_id = sorted(beam_id_data_dict.keys())
        # print(beam_id)

        # get valid beam id
        valid_beam_id = beam_id[:-pred_len]
        valid_beam_id = valid_beam_id[seq_len*2:]
        # print(valid_beam_id)

        # gps_u1 is fixed by scenario
        gps_u1_data_collect = ['./'+scen+'/unit1/GPS_data/gps_location.txt']
        for id in valid_beam_id:
            camera_data_collect = []
            radar_data_collect = []
            lidar_data_collect = []
            gps_u2_data_collect = []
            beam_data_collect = []

            # find available data
            flag = True
            multi_data_id = list(range(id, id -2*seq_len, -2))
            multi_data_id.reverse()
            for data_id in multi_data_id:
                # check in dict or not
                if data_id in camera_id_data_dict and data_id in radar_id_data_dict and data_id in lidar_id_data_dict:
                    camera_data_collect.append('./'+scen+'/unit1/camera_data/'+camera_id_data_dict[data_id])
                    radar_data_collect.append('./'+scen+'/unit1/radar_data/'+radar_id_data_dict[data_id])
                    lidar_data_collect.append('./'+scen+'/unit1/lidar_data/'+lidar_id_data_dict[data_id])
                else:
                    flag = False

            gps_u2_data_id = list(range(id-6, id -6*3, -6))
            gps_u2_data_id.reverse()
            for data_id in gps_u2_data_id:
                # check in dict or not
                if data_id in gps_u2_id_data_dict:
                    gps_u2_data_collect.append('./'+scen+'/unit2/GPS_data/'+gps_u2_id_data_dict[data_id])
                else:
                    flag = False

            beam_index = beam_id.index(id)
            if (beam_id[beam_index+pred_len]-beam_id[beam_index]) < 10:
                beam_data_id = beam_id[beam_index:beam_index+pred_len]
                for data_id in beam_data_id:
                    # check in dict or not
                    if data_id in beam_id_data_dict:
                        beam_data_collect.append('./'+scen+'/unit1/mmWave_data/'+beam_id_data_dict[data_id])
                    else:
                        flag = False
            else:
                flag = False

            if flag:
                # get beam label for mmWave data
                beam_label = get_beam_label(beam_data_collect, RootPath)
                # print(index)
                # print(beam_label)
                # gather all data for one item
                data_collect = [index] + camera_data_collect + radar_data_collect + lidar_data_collect + \
                               gps_u1_data_collect + gps_u2_data_collect + beam_data_collect + beam_label
                # print(data_collect)
                # if index > 3000:
                #     pass

                # add the data into csv
                with open(SaveOutputFile, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(data_collect)
                    index += 1
            else:
                continue
    print(index)
    print(SaveOutputFile)



def createDataset(InputFile, OutputFile, Keyword):
    RawFile = InputFile
    CleanedFile = OutputFile + '.csv'

    if not os.path.exists(CleanedFile):
        with open(CleanedFile, 'w') as file:
            pass

    with open(RawFile) as infile, open(CleanedFile, 'w') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                if Keyword in row[reader.fieldnames[1]]:
                    writer.writerow(row)
            except:
                continue


data_root = './Dataset'
trainval_root = data_root+'/Multi_Modal/'
train_root_csv ='ml_challenge_dev_multi_modal_30to5.csv'
createRootCSV(trainval_root, train_root_csv, seq_len=20, pred_len=5)

data_root = './Dataset'
trainval_root = data_root+'/Multi_Modal/'
train_root_csv ='ml_challenge_dev_multi_modal_30to5.csv'
for keywords in ['scenario32', 'scenario33', 'scenario34']:
    createDataset(trainval_root+train_root_csv, trainval_root+keywords+'_30to5', keywords)
    print(trainval_root + keywords + '_30to5')
