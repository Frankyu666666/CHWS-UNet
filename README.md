This is the official implementation of CHWS-UNet.   
UNet_CHWS.py is the  implementationi of CHWS-UNet on basis of U-Net.  
![CHWS-UNet](UNet_base.png)
#######UNet_CHWS.py#######  
#By default, four CHWS units are inserted into four shortcutpaths between the encoder and decoder as shortcut is set to [1, 1, 1, 1]("1" at position _i_ indicates the CHWS unit is inserted at _i_th stage)  
for i in range(0, len(shortcut)):  
> if shortcut[i]:  
>> setattr(self, f"tp{i + 1}", CHWS(kernel_size=7, in_planes=[64*pow(2, i), 256//(pow(2, i)), 256//(pow(2, i))], mac_pattern=mac_pattern, mic_pattern=mic_pattern))  
