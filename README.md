# autoplex_soap_turbo

## Installation

Here, we depend on my fork of the autoplex repository. 

Hence to make this repository work we must

```bash
git clone --recursive https://github.com/TiganyZ/autoplex_soap_turbo.git
```
```
```

## Setup 

### Python Environment 

Run the uv setup script in the top-level directory 

```bash
bash setup/setup_env_uv.sh
```
```
```

This will create the environment locally on this machine with uv (which is nicer to use than conda/mamba etc).

### Machines

We will rely on doing automated training by multiple machines. These must have *passwordless ssh access*. 

The python environment on each of these machines must be identical. 

Hence, we define the *remote* machines in the config file along with the root directories of which we want to install the environment. 

Here is a sample `config/machines.yaml` so one can setup the calculations

```yaml
databases: autoplex
machines: 
  roihuc:
    rootdir: /scratch/project_2017844/gap_calculations
  roihug:
    rootdir: /scratch/project_2017844/gap_gpu_calculations
  triton:
    rootdir: /scratch/work/zarrout1/potentials/gap_training
```
```
```









