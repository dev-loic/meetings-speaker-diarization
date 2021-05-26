# Sources
- https://github.com/pyannote/pyannote-audio
- https://github.com/pyannote/pyannote-audio-hub#speaker-diarization
- https://colab.research.google.com/github/pyannote/pyannote-audio/blob/master/notebooks/introduction_to_pyannote_audio_speaker_diarization_toolkit.ipynb#scrollTo=f5u8wRm3GYFr

# Data analysis
- Document here the project: meetings-speaker-diarization
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for meetings-speaker-diarization in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/meetings-speaker-diarization`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "meetings-speaker-diarization"
git remote add origin git@github.com:{group}/meetings-speaker-diarization.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
msd-run
```

# Install

Go to `https://github.com/{group}/meetings-speaker-diarization` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/meetings-speaker-diarization.git
cd meetings-speaker-diarization
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
msd-run
```
