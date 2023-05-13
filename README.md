使用 `pip install -r requirements.txt` 安装第三方包

如果有 `cuda` 环境，需要重新安装 `torch`

`pip uninstall torch torchvision`

然后下载 [torch](https://pytorch.org/get-started/locally/)

选择对应的环境获取下载地址

例如：

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装完成后进入`python`

测试
```python
import torch
torch.cuda.is_available()
```

