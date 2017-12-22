The Iceberg classifier is a classification problem where the
competition dataset contains the iceberg and the ship images
data taken by satellite and numerical data for angle of projection
taken for image. The objective is to accurately identify
the iceberg from a ship. The dataset contains 3 input variables
and a binary class variable:
band1: This has 5625 elements, which are the flattened array
for 75 x 75 pixels, as numbers and each number corresponds
to polarization by HH channel(transmit/receive horizontally)
band2: This also has same size of pixels as numbers and the
number corresponds to polarization by HV channel (transmit
horizontally and receive vertically).
Inc angle: The angle represents the angle of projection from
satellite to the object.
Is iceberg: 0 for ship and 1 for iceberg.
The prediction consists of the probability of every image
in the test data of being an iceberg, which in turn quantifies
the goodness of the model.
