import torch
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--target_concept", type=str, required=True, default=None)
parser.add_argument("--eval_path", type=str, required=True)
parser.add_argument("--eval_type", type=str, required=True, help='style or object')
parser.add_argument("--save_path", type=str, required=True)

args = parser.parse_args()

target_concept = args.target_concept

if args.eval_type == 'style':
    eval_path = args.eval_path
    ua_ira_path = os.path.join(eval_path, f'{target_concept}/{target_concept}.pth')
    cru_path = os.path.join(eval_path, f'{target_concept}/{target_concept}-obj.pth')
    i_n = 50
    c_n = 20   
else:
    eval_path = args.eval_path
    ua_ira_path = os.path.join(eval_path, f'{target_concept}/{target_concept}-obj.pth')
    cru_path = os.path.join(eval_path, f'{target_concept}/{target_concept}.pth')
    i_n = 19
    c_n = 51

save_path = args.save_path
save_path = os.path.join(save_path,f'{args.eval_type}')

if not os.path.exists(save_path):
        os.makedirs(save_path)


loaded_results = torch.load(ua_ira_path)
cross_results = torch.load(cru_path)





result = {}
unlean_acc = 0
ira_acc = 0
cra_acc = 0

for keys in loaded_results['acc'].keys():
    ua_ira_acc = loaded_results['acc'][keys]
    if hasattr(ua_ira_acc, 'item'):
        ua_ira_acc = ua_ira_acc.item()
    
    
    if keys == target_concept:
        unlean_acc = ua_ira_acc
        result['unlearn_acc'] =  round((1 - unlean_acc)* 100,2)

    
    else:
        ira_acc += ua_ira_acc


for keys in cross_results['acc'].keys():
    cra_acc_p = cross_results['acc'][keys]
    if hasattr(cra_acc_p, 'item'):
        cra_acc_p = cra_acc_p.item()
    cra_acc += cra_acc_p

    
  

result['in_acc'] = round((ira_acc / i_n) * 100,2)   
result['cross_acc'] = round((cra_acc / c_n) * 100,2)  
      

print(result)
with open(os.path.join(save_path,f'{target_concept}-results.json'), 'w') as f:
        json.dump(result, f) 
        
