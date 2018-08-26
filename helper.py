import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

class Helper:
    def __init__(self):
        self.smart_to_name_data={
            'smart_1_raw': 'Read Error Rate',
            'smart_2_raw': 'Throughput Performance',
            'smart_3_raw': 'Spin-Up Time',
            'smart_4_raw': 'Start/Stop Count',
            'smart_5_raw': 'Reallocated Sectors Count',
            'smart_6_raw': 'Read Channel Margin',
            'smart_7_raw': 'Seek Error Rate',
            'smart_8_raw': 'Seek Time Performance',
            'smart_9_raw': 'Power-On Hours',
            'smart_10_raw': 'Spin Retry Count',
            'smart_11_raw': 'Recalibration Retries',
            'smart_12_raw': 'Power Cycle Count',
            'smart_13_raw': 'Soft Read Error Rate',
            'smart_22_raw': 'Current Helium Level', # only HGST
            'smart_170_raw': 'Available Reserved Space', # see smart_232
            'smart_171_raw': 'SSD Program Fail Count', # only Kingston
            'smart_172_raw': 'SSD Erase Fail Count', # only Kingston
            'smart_173_raw': 'SSD Wear Leveling Count',
            'smart_174_raw': 'Power-off Retract Count',
            'smart_175_raw': 'Power Loss Protection Failure', # Binary Mask
            'smart_176_raw': 'Erase Fail Count',
            'smart_177_raw': 'Wear Range Delta',
            'smart_179_raw': 'Used Reserved Block Count Total', # only samsung
            'smart_180_raw': 'Unused Reserved Block Count Total', # only hp
            'smart_181_raw': 'Program Fail Count Total',
            'smart_182_raw': 'Erase Fail Count',
            'smart_183_raw': 'SATA Downshift Error Count', # only WD, Samsung, Seagate
            'smart_184_raw': 'End-to-End error',
            'smart_185_raw': 'Head Stability', # only WD
            'smart_186_raw': 'Induced Op-Vibration Detection', # only WD
            'smart_187_raw': 'Reported Uncorrectable Errors',
            'smart_188_raw': 'Command Timeout',
            'smart_189_raw': 'High Fly Writes', # most Seagate, some WD
            'smart_190_raw': 'Temperature Difference',
            'smart_191_raw': 'G-sense Error Rate',
            'smart_192_raw': 'Power-off Retract Count',
            'smart_193_raw': 'Load Cycle Count',
            'smart_194_raw': 'Temperature',
            'smart_195_raw': 'Hardware ECC Recovered', # vendor specific
            'smart_196_raw': 'Reallocation Event Count',
            'smart_197_raw': 'Current Pending Sector Count',
            'smart_198_raw': '(Offline) Uncorrectable Sector Count',
            'smart_199_raw': 'UltraDMA CRC Error Count',
            'smart_200_raw': 'Multi-Zone Error Rate',
            'smart_201_raw': 'Soft Read Error Rate',
            'smart_202_raw': 'Data Address Mark errors',
            'smart_203_raw': 'Run Out Cancel',
            'smart_204_raw': 'Soft ECC Correction',
            'smart_205_raw': 'Thermal Asperity Rate',
            'smart_206_raw': 'Flying Height',
            'smart_207_raw': 'Spin High Current',
            'smart_208_raw': 'Spin Buzz',
            'smart_209_raw': 'Offline Seek Performance',
            'smart_210_raw': 'Vibration During Write', # only old maxtor
            'smart_211_raw': 'Vibration During Write',
            'smart_212_raw': 'Shock During Write',
            'smart_220_raw': 'Disk Shift',
            'smart_221_raw': 'G-Sense Error Rate',
            'smart_222_raw': 'Loaded Hours',
            'smart_223_raw': 'Load/Unload Retry Count',
            'smart_224_raw': 'Load Friction',
            'smart_225_raw': 'Load/Unload Cycle Count',
            'smart_226_raw': "Load 'In'-time",
            'smart_227_raw': 'Torque Amplification Count',
            'smart_228_raw': 'Power-Off Retract Cycle',
            'smart_230_raw': 'GMR Head Amplitude', # only hddd
            'smart_231_raw': 'Temperature', # only hdd
            'smart_232_raw': 'Endurance Remaining', #only ssd
            'smart_233_raw': 'Power-On Hours', # only hdd
            'smart_234_raw': 'Average erase count AND Maximum Erase Count', # Bitmask
            'smart_235_raw': 'Good Block Count AND System(Free) Block Count', # Bitmask
            'smart_240_raw': 'Head Flying Hours', # not fujitsu
            'smart_241_raw': 'Total LBAs Written',
            'smart_242_raw': 'Total LBAs Read',
            'smart_243_raw': 'Total LBAs Written Expanded', # conneted to 241
            'smart_244_raw': 'Total LBAs Read Expanded', # connected to 242
            'smart_249_raw': 'NAND Writes (1GiB)',
            'smart_250_raw': 'Read Error Retry Rate',
            'smart_251_raw': 'Minimum Spares Remaining',
            'smart_252_raw': 'Newly Added Bad Flash Block',
            'smart_254_raw': 'Free Fall Protection'
        }
        
        self.all_possible_columns = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']
        for i in range(0, 256):
            self.all_possible_columns.append('smart_{}_raw'.format(i+1))

    def smart_to_name(self):
        return self.smart_to_name_data

    def column_list(self):
        return self.all_possible_columns

    def extract_smart_values(string):
        """
        Extract Smart values from a given string. We are only interisted in the `raw` S.M.A.R.T. values and the first columns.
        """
        parts = string.split(',')
        ids = {0:0,1:1,2:2,3:3,4:4}

        for k, p in enumerate(parts):
            if 'raw' in p:
                v = int(p.replace('smart_','').replace('_raw',''))
                ids[k] = v+4
        return ids


    def fill_content(m, parts):
        """
        This function add empty space to a given list, to make sure that several of this list have a consistend filled csv as output
        """
        content = list(range(0,260))
        # fill with empty string
        for i in range(0, 260):
            content[i] = ''

        for i in range(0,5):
            content[i] = str(parts[i])

        for column, smart_value in m.items():
            content[smart_value] = str(parts[column])
        return content
    
    def plot_by_algorithm(df):
        all_clfs = []
        for algo in df['clf_name'].unique():
            f_beta = df[df['clf_name'] == algo]['f_beta_score']
            all_clfs.append(go.Histogram(x=f_beta, xbins=dict(start=0,end=1,size=0.05), name='{} - {:.3f}'.format(algo, f_beta.mean())))

        plotly.offline.iplot({
            'data': all_clfs,
            "layout": go.Layout(barmode='stack', title="Challenging all Classifiers against each other")
        })

    def plot_by_model(df):
        all_clfs = []
        for drive in df['drive'].unique():
            f_beta = df[df['drive'] == drive]['f_beta_score']
            all_clfs.append(go.Histogram(x=f_beta, xbins=dict(start=0,end=1,size=0.05), name='{} - {:.3f}'.format(drive.replace('.csv', ''), f_beta.mean())))

        plotly.offline.iplot({
            'data': all_clfs,
            "layout": go.Layout(barmode='stack', title="Challenging all drive models against each other")
        })
        
    def plot_by_manufacturer(df):
        df['manufacturer'] = df['drive'].apply(Helper.__map_manufactor)
        all_clfs = []
        for manufactur in df['manufacturer'].unique():
            f_beta = df[df['manufacturer'] == manufactur]['f_beta_score']
            all_clfs.append(go.Histogram(x=f_beta, xbins=dict(start=0,end=1,size=0.05), name='{} - {:.3f}'.format(manufactur, f_beta.mean())))

        plotly.offline.iplot({
            'data': all_clfs,
            'layout': go.Layout(barmode='stack', title="Challenging all manufacturs against each other")
        })
        
        
    def __map_manufactor(s):
        manufactur_map = {
            'toshiba': 'Toshiba',
            'wdc': 'Western Digital', # https://de.wikipedia.org/wiki/Western_Digital
            'hgst': 'Hitachi Global Storage Technologies Ltd.', # was bought by western digital 2012, see: https://de.wikipedia.org/wiki/HGST
            'st': 'Seagate' # https://de.wikipedia.org/wiki/Seagate_Technology
        }
        for k, v in manufactur_map.items(): 
            if k in s.lower(): 
                return v
    