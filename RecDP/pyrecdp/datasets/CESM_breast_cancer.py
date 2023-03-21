from .base_api import base_api

def try_finall(regex, input_str):
    import re
    ret = re.findall(regex, input_str)
    if len(ret) == 0:
        return None
    else:
        return ret[0]
    
def extract_info_from_report(report_content):
    method = None
    side = None
    p_id = None
    ret = {}
    lines = [i for i in report_content.split('\n') if i != '']
    for t in lines:
        if 'PATIENT NO.' in t:
            p_id = try_finall("\d+", t)
            method = None
            side = None                    
        elif 'SOFT TISSUE MAMMOGRAPHY REVEALED:' in t:
            method = 'DM'
        elif 'OPINION:' in t:
            method = 'OP'
        elif 'CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:' in t:
            method = 'CESM'
        elif 'Right Breast' in t:
            side = "R"
        elif 'Left Breast' in t: 
            side = "L"
        else:
            if side is None or method is None:
                continue
            cur_key = f"{side}_{method}"
            if cur_key not in ret:
                ret[cur_key] = {}
                ret[cur_key]['Side'] = side
                ret[cur_key]['Patient_ID'] = int(p_id)
                ret[cur_key]['Type'] = method
                ret[cur_key]['symptoms'] = t
    
    return ret

class CESM_breast_cancer(base_api):
    def __init__(self, scale = 'full'):
        super().__init__()
        file_list = {
            'medical_report': "https://wiki.cancerimagingarchive.net/download/attachments/109379611/Medical%20reports%20for%20cases%20.zip?api=v2",
            'manual_annotations': "https://wiki.cancerimagingarchive.net/download/attachments/109379611/Radiology%20manual%20annotations.xlsx?api=v2"
        }

        self.saved_path = dict()
        self.saved_path['medical_report'] = self.download_url("Medical reports for cases", file_list['medical_report'], unzip = True)
        self.saved_path['manual_annotations'] = self.download_url("radiology_manual_annotations.xlsx", file_list['manual_annotations'])

    def to_pandas(self, nrows = None):
        import pandas as pd
        import os, docx2txt
        ret = {}
        ret['manual_annotations'] = pd.read_excel(self.saved_path['manual_annotations'], sheet_name="all")
        #ret['medical_report'] = {}
        medical_report_content = []
        medical_extracted = []
        for f in os.listdir(self.saved_path['medical_report']):
            try:
                file_content = docx2txt.process(os.path.join(self.saved_path['medical_report'], f))
            except Exception as e:
                file_content = ""
                Warning(e)
            medical_report_content.append(file_content)
            ext = extract_info_from_report(file_content)
            for side, line in ext.items():
                medical_extracted.append(line)

        ret['medical_report'] = pd.DataFrame.from_records(medical_extracted)
        
        return ret
         