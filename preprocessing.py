# As suggested by the first review, I moved the code to this python script.
import os
from tqdm import tqdm
import requests
import zipfile
import glob
import shutil
from helper import Helper
class Preprocessing():
    def __init__(self):
        self.base_path = 'https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data'
        os.makedirs('raw', exist_ok=True)      
        os.makedirs('drives', exist_ok=True)   
        os.makedirs('train', exist_ok=True)    
        os.makedirs('test', exist_ok=True)   
        os.makedirs('validate', exist_ok=True)
        os.makedirs('tmp', exist_ok=True)
        os.makedirs('drives_minified', exist_ok=True)
        os.makedirs('sklearn_models', exist_ok=True)
        os.makedirs('keras_models', exist_ok=True)
        os.makedirs('xgboost_models', exist_ok=True)
        os.makedirs('lightgbm_models', exist_ok=True)
        
        # raw - the filename of the raw data
        # url - the url that the data have to be fetched from
        # zip - argument for the extraction, as the raw files for 2013, 2014 and 2015 contain a directory inside. All others are just the files directly, so we have to provide a directory to extract them. 
        self.records = [
            {'raw': os.path.join('raw', 'data_Q2_2018.zip'), 'url': '{}/data_Q2_2018.zip'.format(self.base_path), 'zip': 'data_Q2_2018'},
            {'raw': os.path.join('raw', 'data_Q1_2018.zip'), 'url': '{}/data_Q1_2018.zip'.format(self.base_path), 'zip': 'data_Q1_2018'},
            {'raw': os.path.join('raw', 'data_Q4_2017.zip'), 'url': '{}/data_Q4_2017.zip'.format(self.base_path), 'zip': 'data_Q4_2017'},
            {'raw': os.path.join('raw', 'data_Q3_2017.zip'), 'url': '{}/data_Q3_2017.zip'.format(self.base_path), 'zip': 'data_Q3_2017'},
            {'raw': os.path.join('raw', 'data_Q2_2017.zip'), 'url': '{}/data_Q2_2017.zip'.format(self.base_path), 'zip': 'data_Q2_2017'},
            {'raw': os.path.join('raw', 'data_Q1_2017.zip'), 'url': '{}/data_Q2_2017.zip'.format(self.base_path), 'zip': 'data_Q1_2017'},
            {'raw': os.path.join('raw', 'data_Q4_2016.zip'), 'url': '{}/data_Q4_2017.zip'.format(self.base_path), 'zip': 'data_Q4_2016'},
            {'raw': os.path.join('raw', 'data_Q3_2016.zip'), 'url': '{}/data_Q3_2017.zip'.format(self.base_path), 'zip': 'data_Q3_2016'},
            {'raw': os.path.join('raw', 'data_Q2_2016.zip'), 'url': '{}/data_Q2_2017.zip'.format(self.base_path), 'zip': 'data_Q2_2016'},
            {'raw': os.path.join('raw', 'data_Q1_2016.zip'), 'url': '{}/data_Q2_2017.zip'.format(self.base_path), 'zip': 'data_Q1_2016'},
            {'raw': os.path.join('raw', 'data_2015.zip'), 'url': '{}/data_2015.zip'.format(self.base_path), 'zip': '2015'},
            {'raw': os.path.join('raw', 'data_2014.zip'), 'url': '{}/data_2014.zip'.format(self.base_path), 'zip': '2014'},
            {'raw': os.path.join('raw', 'data_2013.zip'), 'url': '{}/data_2013.zip'.format(self.base_path), 'zip': '2013'}]
        
    def download(self):
        with tqdm(total=len(self.records)) as pbar:
            for record in self.records:
                full_local_path = record['raw']
                full_url = record['url']

                if not os.path.exists(full_local_path):
                    pbar.set_description('Downloading {}'.format(full_url))

                    r = requests.get(full_url, stream=True)
                    with open(full_local_path, 'wb') as out_file:
                        shutil.copyfileobj(r.raw, out_file)

                pbar.update(1)
                
    def unpack(self):
        with tqdm(total=len(self.records)) as pbar:
            for record in self.records:
                full_local_path = record['raw']
                full_url = record['url']

                if not os.path.exists(record['zip']) and not os.path.exists('data_{}'.format(record['zip'])):

                    pbar.set_description('Unpacking {}'.format(full_local_path))
                    zip_ref = zipfile.ZipFile(full_local_path, 'r')

                    # sometimes the zip have a directory as root, sometimes files directly
                    if not 'data' in record['zip']:
                        zip_ref.extractall()

                    else:
                        os.makedirs(record['zip'], exist_ok=True)
                        zip_ref.extractall(record['zip'])           
                    zip_ref.close() 

                pbar.update(1)
                
    def rename(self):
        # Rename year
        for year in [2013, 2014, 2015]:
            if not os.path.exists('data_{}'.format(year)):
                os.rename(str(year), 'data_{}'.format(year))
                
    def split_by_drive(self):
        if not os.path.exists('qualified.csv'): # this is our hint, that we don't need to rerun the preprocessing agian
            file_handler = {}
            total_files = glob.glob(os.path.join("data*","*.csv"), recursive=True)
            with tqdm(total=len(total_files)) as pbar:
                for file in total_files:
                    with open(file, 'rt') as f:
                        ids = []
                        for line in f:
                            if 'date' in line:
                                ids = Helper.extract_smart_values(line)
                            else:
                                parts = line.split(',')
                                model = parts[2]
                                content = Helper.fill_content(ids, parts)
                                if model in file_handler:
                                    file_handler[model].write(",".join(content))
                                else:
                                    file_handler[model] = open(os.path.join('tmp', '{}.csv'.format(model)), 'at')
                                    file_handler[model].write(",".join(content))

                    pbar.update(1)

            # close all file handlers
            for handler in file_handler.values():
                handler.close()
            
    def split_by_drive_move(self):
        total_files = glob.glob(os.path.join('tmp','*.csv'))
        with tqdm(total=len(total_files)) as pbar:
            for file in total_files:
                csv_name = file.split(os.sep)[-1]
                target = os.path.join('drives', csv_name)
                shutil.move(file, target)
                pbar.update(1)
                
    def sorted_max(self, smarts):
        """
        Input: dict with key = smart value "column", value = number of appearence of this smart value in the dataset.
        Output: all keys that have a max appearence in the entire dataset for this drive. 
        This will make sure that the data is trainable by an algorithm, containing no "nan"s
        """
        current_max = 0
        for k, v in smarts.items():
            # make sure to not count the "failure" column as smart value
            if k > 4: 
                current_max = max(v, current_max)

        # return only values with max value
        result = []
        for k, v in smarts.items():
            if v == current_max and k > 4:
                result.append(k)
        return sorted(result)

    def minify(self, file):
        """
        Minify passed file. Minify will:
        * count the amount a smart value was present in the file
        * this is used to get the "important" or "fully filled" features of the drives
        * remove everything else from the file
        """
        # Basic smart values that exist from the beginning of the dataset, shifed by 4, see below.
        allowed_smart_value_index = [ 4,   5,   6,   7,   8,   9,  11,  12,  13,  14,  15,  16,  17,  19,
           187, 188, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 227, 229, 
           244, 245, 246, 254, 255, 256, 258]
        csv_name = file.split(os.sep)[-1]
        tmp_path = os.path.join('tmp', csv_name)
        target_path = os.path.join('drives_minified', csv_name)
        if not os.path.exists(target_path):

            # Calculate all smart values that are present in the dataset for this specific drive
            smarts = {}
            with open(file, 'rt') as f:
                for line in f:
                    parts = line.split(',')
                    for i, p in enumerate(parts):
                        if len(p) > 0 and i in allowed_smart_value_index:
                            if i in smarts:
                                smarts[i] = smarts[i] + 1
                            else:
                                smarts[i] = 1

            save_file_handler = open(tmp_path, 'at')
            # write header
            # i-4 makes sure that the parts map correctly to the smart values
            minified_header = self.sorted_max(smarts)
            header_items = ['failure'] + ['smart_{}_raw'.format(i-4) for i in minified_header]
            save_file_handler.write(",".join(header_items) + '\n')
            with open(file, 'rt') as f:
                for line in f:
                    minified = []

                    parts = line.split(',')
                    for i, p in enumerate(parts):                  
                        # We don't need date, serial_numer, model and capacity_bytes - so we skip index 0,1,2,3
                        # Value 4 indicate the failure
                        # Value [5:] Indicate the smart value from 1 ongoing
                        if i in allowed_smart_value_index and i in minified_header and len(p)>0:
                            minified.append(p)

                        if i == 4:
                            minified.append(p)
                    save_file_handler.write(",".join(minified) + "\n")
            save_file_handler.close()
            shutil.move(tmp_path, target_path)
        
    def drive_metrics(self, file):
        """
        Calculate the metadata of a minified file.
        returns a dict with
        * "file" - name of the file
        * "failure" - the amount of failures as int
        * "ok" the amount of "ok" records as int
        * "lines" - number of lines to check if some algorithms perform better with less or more data
        * train - number of failures that can be used for training
        * test - number of failures that can be used for test
        * validate - number of failures that can be used for the validation set
        """
        if len(file) == 0:
            return {'failure':0}
        fails = 0
        ok = 0
        lines = 0
        size = os.stat(file).st_size
        with open(file, 'rt') as f:
            for line in f:
                lines += 1
                # Ignore header
                if not 'failure' in line:
                    parts = line.split(',')

                    # Ignore final newline
                    if parts[0] != '\n':
                        failure = int(parts[0])
                        if failure == 1:
                            fails += 1
                        else:
                            ok += 1
        if fails <= 5:
            train =  fails - 2
            test = 1
            validate = 1
        else:
            train =  floor(fails * 0.8)
            test = floor((fails - train) / 2)
            validate = floor((fails - train) / 2)

        return {
            'file': file,
            'failure':fails,
            'ok': ok,
            'drive_csv': file.split(os.sep)[-1],
            'size': size,
            'lines': lines,
            'train': train,
            'test': test,
            'validate': validate,
        }
    
    def final_split(row):
        train = row['train']
        test = row['test']
        validate = row['validate']
        model = row['drive_csv']
        file = os.path.join('drives_minified', model)

        path_train_tmp = os.path.join('tmp', 'train_{}'.format(model))
        path_test_tmp = os.path.join('tmp', 'test_{}'.format(model))    
        path_validate_tmp = os.path.join('tmp', 'validate_{}'.format(model))      

        path_train = os.path.join('train', model)
        path_test = os.path.join('test', model)
        path_validate = os.path.join('validate', model)


        if not os.path.exists(path_train) and not os.path.exists(path_test) and not os.path.exists(path_validate):
            file_handler_train =  open(path_train_tmp, 'a+')
            file_handler_test = open(path_test_tmp, 'a+')
            file_handler_validate = open(path_validate_tmp, 'a+')                        

            with open(file, 'rt') as f:              
                for line_index, line in enumerate(f):
                    try: 
                        # Ignore empty newline
                        if line != '\n':

                            # Handling header, write it everywhere
                            if line_index == 0:
                                file_handler_train.write(line)
                                file_handler_test.write(line)
                                file_handler_validate.write(line)
                            elif 'failure' in line:
                                # ignore if double header exist as split in parallel and multiple writes with header accured
                                continue
                            else:
                                parts = line.split(',')
                                failure = int(parts[0])

                                # Can we use random split?
                                if failure == 0:
                                    choise = random.random()
                                    if choise >= 0.0 and choise <=0.8:
                                        # use for training
                                        file_handler_train.write(line)

                                    elif choise > 0.8 and choise <= 0.9:
                                        # use for testing
                                        file_handler_test.write(line)

                                    else:
                                        # use for validation
                                        file_handler_validate.write(line)
                                # We have to do an explicit split according to the values previously calculated.
                                else:
                                    if train > 0:
                                        train -= 1
                                        file_handler_train.write(line)
                                    elif test > 0:
                                        test -= 1
                                        file_handler_test.write(line)
                                    elif validate > 0:
                                        validate -= 1
                                        file_handler_validate.write(line)
                    except:
                        raise Exception('index: {}, line: {}, file: {}'.format(line_index, line, model))


            file_handler_train.close()
            file_handler_test.close()
            file_handler_validate.close()

            # Make sure that the file handler is closed
            # sleep(1000)
            shutil.move(path_train_tmp, path_train)
            shutil.move(path_test_tmp, path_test)
            shutil.move(path_validate_tmp, path_validate)