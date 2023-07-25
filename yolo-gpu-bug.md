# Traning TOLO on GPU bug

I created a new envirionemt with only `ultralytics` and `ipykernel` installed.

After running this cell:

```
import ultralytics
from ultralytics import YOLO
import torch
```

...I got this warrning:

```
WARNING  Ultralytics settings reset to default values. This may be due to a possible problem with your settings or a recent ultralytics package update.
View settings with 'yolo settings' or at 'C:\Users\Admin\AppData\Roaming\Ultralytics\settings.yaml'
Update settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'.
```

> **From GPT-3**
> The warning provides you with some information on how to proceed:
>
> To view the current settings, you can use the command yolo settings in your terminal or command prompt.
>
> If you want to **update the settings** to your preferred values, you can use the command yolo settings **key=value**, where you replace key with the specific setting you want to change and value with the new value you want to set.
>
> For example, if you want to change the runs_dir setting to a different directory, you would use the command: yolo settings runs_dir=path/to/dir
>
> By checking the settings and updating them if necessary, you can ensure that YOLO functions as expected with the desired configurations.

I decided to reinstall cuDNN. From [this](https://developer.nvidia.com/rdp/cudnn-download) page.
And that didn't helped.

### Manual installation of `torch`.

The `torch` library is automatically installed with `ultralytic`. However, it didn't saw my GPU. I tried to install torch manually [following the documentation](https://pytorch.org/get-started/locally/). As the result `torch.cuda.is_available()` finally returned `True`! But only for one run...

### Changeing `pip` to `conda`

So, I created a conda envirionment...

`conda install ultralytics` didn't worked.

I installed `ulatralytics` with this command:

```
conda install -c conda-forge ultralytics
```

## XD
ImportError: DLL load failed while importing cv2: Nie można odnaleźć określonego modułu.