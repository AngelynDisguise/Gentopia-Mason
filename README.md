# Gentopia-Mason

**IMPORTANT NOTICE: This code repository was adapted from [Gentopia.AI](https://github.com/Gentopia-AI) to support Mason Activities.** 

Authors: Ziyu Yao (ziyuyao@gmu.edu), Saurabh Srivastava (ssrivas6@gmu.edu), and Murong Yue (myue@gmu.edu)

Copyright and license should go to Gentopia.AI.

**Windows Users: We currently only support Powershell.**

## Installation 💻
First, clone this repository:
```
git clone git@github.com:LittleYUYU/Gentopia-Mason.git
cd Gentopia-Mason
```
If you have not set up your ssh keys yet, you may receive an error. Instead, you can clone the repository using the following command:
```
git clone https://github.com/LittleYUYU/Gentopia-Mason.git
```

Next, we will create a virtual environment and install the library:
```
conda create --name gentenv python=3.10
conda activate gentenv
pip install -r requirements.txt
```

Most of the agent construction and execution activities will happen within `GentPool`. For the `gentopia` library to be used within `GentPool`, set up the global environment:

**For Linux/Mac**
```
export PYTHONPATH="$PWD/Gentopia:$PYTHONPATH"
```
**For Windows (Windows Powershell)**: 
```
$env:PYTHONPATH = "$PWD/Gentopia;$env:PYTHONPATH"
```

In addition, since we will be using OpenAI's API, we also need to create a `.env` file under `GentPool` and put the API Key inside. The key will be registered as environmental variables at run time. To do this,
first, let's change the directory:
```
cd GentPool
```

**For Linux/Mac**
```
touch .env
echo "OPENAI_API_KEY=<your_openai_api_key>" >> .env
```

**For Windows (Windows Powershell)**
```
$env:OPENAI_API_KEY="<YOUR KEY>"
```
Now you are all set! Let's create your first Gentopia Agent.


## Quick Start: Clone a Anthropic Claude Agent
GentPool has provided multiple template LLM agents. To get started, we will clone the "vanilla agent" from `GentPool/gentpool/pool/anthropic_template` with the following command:

**For Linux/Mac**
```
./clone_agent anthropic_template <your_agent_name> 
```

**For Windows (Windows Powershell)**
```
.\clone_agent.bat anthropic_template <your_agent_name> #note the .bat file 
```

This command will initiate an agent template under `./GentPool/gentpool/pool/<your_agent_name>`. The agent configuration can be found in `./GentPool/gentpool/pool/<your_agent_name>/agent.yaml` (note the agent type `anthropic_template`). The vanilla prompt it uses can be found in the source code of `Gentopia`; see `./Gentopia/gentopia/prompt/vanilla.py`.

You can now run your agent via:
```
python assemble.py <your_agent_name>
```
This vanilla agent simply sends the received user query to the backend LLM and returns its output. Therefore, for many complicated tasks, such as those requiring accessing the latest materials, it will fail. 

## Getting started with Ramsay Agent
Similar to the `rewoo` agent "elon" from the Gentopia tutorials, another `rewoo` agent "ramsay" is here to provide professional recipes based on your personal needs.

**To run it:**
```
python assemble.py ramsay
```



