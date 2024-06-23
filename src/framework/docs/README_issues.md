# Known issues

1. Data transmission bottleneck

   Currently VFLAIR supports a maximum transmission of about 1GB, and greater data transmission optimization is in progress

2. Saving trained models

    Currently only supports Qwen2 model saving in distributed mode.

3. modify config.py when using Qwen2
   
    Change the line 31 to `self.split_index = kwargs.get('split_index', (2,-2))` in config.py when you use QWen2 model, and change it back if you use other models. 
