# RandomAgent

If running headless, execute:

    xvfb-run -s "-screen 0 1400x900x24" python random_agent.py -b /tmp/random-agent

And then run:

    python uploader.py -b /tmp/random-agent -w https://gist.github.com/gdb/62d8d8f5e13270d4b116336ae61240db -a random-v3
