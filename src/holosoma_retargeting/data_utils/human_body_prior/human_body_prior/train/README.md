# Train VPoser from Scratch
To train your own VPoser with new configuration duplicate the provided **V02_05** folder while setting a new experiment ID 
and change the settings as you desire. 
First you would need to download the 
[AMASS](https://amass.is.tue.mpg.de/) dataset, then following the [data preparation tutorial](../data/README.md)
prepare the data for training. 
Following is a code snippet for training that can be found in the [example training experiment](https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/train/V02_05/V02_05.py):

```python
import glob
import os.path as osp

from human_body_prior.tools.configurations import load_config
from human_body_prior.train.vposer_trainer import train_vposer_once

def main():
    expr_id = 'V02_05'

    default_ps_fname = glob.glob(osp.join(osp.dirname(__file__), '*.yaml'))[0]

    vp_ps = load_config(default_ps_fname)

    vp_ps.train_parms.batch_size = 128

    vp_ps.general.expr_id = expr_id

    total_jobs = []
    total_jobs.append(vp_ps.toDict().copy())

    print('#training_jobs to be done: {}'.format(len(total_jobs)))
    if len(total_jobs) == 0:
        print('No jobs to be done')
        return

    for job in total_jobs:
        train_vposer_once(job)
``` 
The above code uses yaml configuration files to handle experiment settings. 
It loads the default settings in *<expr_id>.yaml* and overloads it with your new args. 

The training code, will dump a log file along with tensorboard readable events file.