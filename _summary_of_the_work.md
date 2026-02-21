# What is going on

we started with the place where we have gotten to so far. the gold dataset is the one that we have generated so that the previous issues are fixed.

# Issue 01:
we trainign it on the pre trained weights. didn't work. around 90 degree error.

# fix 01:
auditing the dataset. 
![initial audit](image.png)

# Issue 02: 
I didnot include the names of the models taht were used for testing. cant see what the hell is it. 

# Fix 02: 
named auditing is done. 

# Issue 03:
the models are too confuding so i cheesed a specific one and asked it to apply the rortations. 

# fix 03:
so the dataste is correct it seems now, the only issue that i can see is the model. the reason is that i am using the weights of the pretrained model which was on video data. what i am going to do is revert back to the old model which is based on the resnet18 but no weights. 