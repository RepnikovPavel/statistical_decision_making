import imageio
# https://ezgif.com/gif-to-mp4/ezgif-1-0d8c4320a75d.gif  best site
with imageio.get_writer("D:\saved_gifs\\random_distrib_with_reg_with_ro_multiple_points_4.gif", mode='I') as writer:
        for i in range(200):
            print(i)
            image = imageio.imread("D:\\saved_fig\\" + str(i) + ".png")
            writer.append_data(image)


# import imageio
# # https://ezgif.com/gif-to-mp4/ezgif-1-0d8c4320a75d.gif  best site
# with imageio.get_writer("C:\\wow_saved_fig\\silvana.gif", mode='I') as writer:
#         for i in range(9):
#             print(i)
#             image = imageio.imread("C:\\wow_saved_fig\\" + str(i+1) + ".png")
#             writer.append_data(image)


