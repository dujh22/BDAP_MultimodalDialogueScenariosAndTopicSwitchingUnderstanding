import json

def reformat(identifier):
    """
    -> {'clip_id':{'dialog':[], 'scene':[], 'session':[]}}
    """
    with open('inputs/{}.json'.format(identifier)) as jh:
        dataset = json.load(jh)
    reformat_dct = {}
    for epi, clip_lst in dataset.items():
        for clip_dct in clip_lst:
            dialog = []
            scene = []
            session = []
            vid = clip_dct['vid']
            for sub in clip_dct['subs']:
                dialog.append(sub['en_text'])
                scene.append(sub['scene'])
                session.append(sub['session'])
            reformat_dct[vid] = {'dialog':dialog, 'session':session, 'scene':scene}
    with open(f'inputs/preprocessed/{identifier}.json', 'w', encoding='utf8') as jh:
        json.dump(reformat_dct, jh)
    
if __name__ == '__main__':
    reformat('MDSS_train')
    reformat('MDSS_valid')
    reformat('MDSS_test')
