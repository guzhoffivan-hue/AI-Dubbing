#!/usr/bin/env python3
"""Select restrained ambience bed candidates from the indexed local library."""
import re
from common import ROOT, WORK, read_json, write_json

def main():
    index=read_json(ROOT/'work'/'sfx_index.json'); files=index['files']
    lines=read_json(WORK/'translation.json')['lines']; context=' '.join((x.get('translated_text','')+' '+x.get('source_text','')).lower() for x in lines)
    rules={'street':['улиц','город','дорог','street','city','traffic'],'office':['офис','рабоч','office'],'restaurant':['ресторан','кафе','ужин','restaurant','bar'],'rain':['дожд','rain'],'wind':['ветер','wind'],'night':['ноч','night'],'crowd':['толп','зрител','crowd','audience'],'nature':['лес','птиц','природ','forest','birds','nature']}
    chosen=None; matched=None
    for category,terms in rules.items():
        if not any(term in context for term in terms): continue
        pool=[x for x in files if category in x['category'] and x.get('duration') and x['duration']>=30]
        if pool: chosen=max(pool,key=lambda x:x['duration']); matched=category; break
    # Do not guess an environment while scene analysis is unavailable.
    write_json(WORK/'ambience.json',{'enabled':bool(chosen),'candidate':chosen,'category':matched,'volume':0.08 if chosen else 0.0,'note':'Selected only on explicit environment cues; no original ambience is used.'})
    print('Selected ambience candidate' if chosen else 'No long ambience candidate found')
if __name__=='__main__': main()
