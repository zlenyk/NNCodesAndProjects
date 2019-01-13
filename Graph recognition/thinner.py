import numpy as np

class Thinner:

    @staticmethod
    def _thinning_iteration(image, iteration):
        marker = np.zeros_like(image)
        zero_ind = np.asarray(np.where(image == 0)).T
        for ind in zero_ind:
            # 0 - white
            # 1 - black
            p2 = 1 - min(image[ind[0]-1][ind[1]], 1)
            p3 = 1 - min(image[ind[0]-1][ind[1]+1], 1)
            p4 = 1 - min(image[ind[0]][ind[1]+1], 1)
            p5 = 1 - min(image[ind[0]+1][ind[1]+1], 1)
            p6 = 1 - min(image[ind[0]+1][ind[1]], 1)
            p7 = 1 - min(image[ind[0]+1][ind[1]-1], 1)
            p8 = 1 - min(image[ind[0]][ind[1]-1], 1)
            p9 = 1 - min(image[ind[0]-1][ind[1]-1], 1)

            transitions =   int(p2 == 0 and p3 == 1) + \
                            int(p3 == 0 and p4 == 1) + \
                            int(p4 == 0 and p5 == 1) + \
                            int(p5 == 0 and p6 == 1) + \
                            int(p6 == 0 and p7 == 1) + \
                            int(p7 == 0 and p8 == 1) + \
                            int(p8 == 0 and p9 == 1) + \
                            int(p9 == 0 and p2 == 1)
            B = p2+p3+p4+p5+p6+p7+p8+p9
            m1 = m2 = 0
            if iteration == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8

            if transitions == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0:
                marker[ind[0]][ind[1]] = 255
        image = np.add(image, marker)
        return image

    @staticmethod
    def thin_image(image):
        thinned_img = image
        while True:
            thinner_img = Thinner._thinning_iteration(thinned_img, 0)
            thinner_img = Thinner._thinning_iteration(thinned_img, 1)
            diff = (thinner_img - thinned_img) / 255
            if np.sum(diff) == 0:
                break
            thinned_img = thinner_img
        return thinned_img
