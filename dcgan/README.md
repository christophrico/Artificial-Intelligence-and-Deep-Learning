## KERAS-DCGAN

*Please note this has been forked from https://github.com/jacobgil/keras-dcgan and modified to fit my needs*




This assumes theano ordering.

---

**Training:**
`python dcgan.py --mode train --batch_size <batch_size>`
python dcgan.py --mode train --path ~/images --batch_size 128

**Image generation:**
`python dcgan.py --mode generate --batch_size <batch_size>`



`python dcgan.py --mode activations`

---
## Result


![generated_image.png](./assets/generated_image.png)



![training_process.gif](./assets/training_process.gif)

---