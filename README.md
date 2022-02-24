# build-a-new-approach-cnn
image_input = Input(shape = (32, 32, 3))
one = Conv2D(32, kernel_size=(3,3), activation='relu' )(image_input)
one = MaxPooling2D((2,2))(one)


one = Conv2D(32, kernel_size=(3,3), activation='relu' )(one)
one = MaxPooling2D((2,2))(one)

one = Flatten()(one)

one = Dense(64,activation='relu')(one)
one = Dense(10,activation='sigmoid')(one)
model = Model(image_input, one)





model.summary()
