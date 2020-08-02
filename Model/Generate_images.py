
def generate_images(model, test_input, tar,option=0):
  """ Check the base path and make dir if needed.
  if not os.path.exists(base_path):
    os.makedirs(base_path)"""
  if option==1:
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,8))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image',  'Predicted Image']

    for i in range(2):
      plt.subplot(1, 2, i+1)
      plt.title(title[i])
      # getting the pixel values between [0, 1] to plot it.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    #plt.savefig(base_path + '/result_{}.png'.format(epoch))
    plt.show()



  else:
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,8))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # getting the pixel values between [0, 1] to plot it.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    #plt.savefig(base_path + '/result_{}.png'.format(epoch))
    plt.show()

