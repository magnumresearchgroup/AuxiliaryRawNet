from datasets.preprocess import get_cm_protocols, get_dataset_annotation, random_split_train_dev, create_non_label_eval_json
import pathlib
import json

if __name__ == '__main__':


    # TODO: MAKE THIS PARSE ARGUMENTS
    print('----Start to Process Data -----')



    args = {}
    args['data_type'] = ['labeled','unlabeled'][0]

    if args['data_type'] == 'labeled':
        print('Start to process labeled data:')
        pathlib.Path('processed_data').mkdir(parents=True, exist_ok=True)
        LA_PRO_DIR = '../data/LA/ASVspoof2019_LA_cm_protocols'
        PRO_FILES = ('ASVspoof2019.LA.cm.train.trn.txt',
                     'ASVspoof2019.LA.cm.dev.trl.txt',
                     'ASVspoof2019.LA.cm.eval.trl.txt')
        SAVE_DIR = '2021_data/'
        DATA_DIR = '../data/'
        split_features= get_cm_protocols(pro_dir=LA_PRO_DIR,
                                         pro_files=PRO_FILES
                                         )
        get_dataset_annotation(split_features,
                               data_dir=DATA_DIR,
                               save_dir=SAVE_DIR,
                               )
        random_split_train_dev()
    else:
        print('TODO: ')



