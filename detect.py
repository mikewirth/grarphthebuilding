import os
import sys
import cv2
import numpy as np

MAX_DIST = 32**2
MATCHING_THRESHOLD = 0.85
DIST_THRESHOLD = 16**2

type_label_map = {
    'Cns': 'C',
    'Gen': 'G',
    'Junction': 'J',
    'Terminal': 'T',
    'TSen': 'Ts',
    'PSen': 'Ps',
    'VflSen': 'Fs',
    'TMon': 'Tm',
    'VflMon': 'Fm',
    'Pu': 'P',
    'Vlv': 'V',
}


class GraphGenerator():
    def __init__(self, img):
        self.graph = self.generate_graph_from_image(img)

    def segment_lines(self, img):
        """ returns line segments, lines are not guaranteed to be unique """
        # grey image and mask lines
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_grey[img_grey == 255] = 0

        kernel = np.ones((15, 15), np.uint8)
        img_grey = cv2.erode(img_grey, kernel)
        # img_grey = cv2.dilate(img_grey, kernel)
        # cv2.imshow('Image', img_grey3)

        # lines = cv2.HoughLinesP(img_grey, 1, np.pi/180, 30, 30)
        # print(lines)
        # for l in lines:
        #     x1,y1,x2,y2 = l[0]
        #     print(x1,y1)
        #     cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(img_grey)

        result_img = fld.drawSegments(img_grey, lines)

        return [l[0].tolist() for l in lines]

    def segment_symbols(self, img):
        """ returns bounding boxes around symbols """
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_grey = (255 - img_grey)

        img_grey[img_grey < 200] = 0

        contours, h = cv2.findContours(
            img_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

        boxes = []
        for c in contours:
            [X, Y, W, H] = cv2.boundingRect(c)

            cv2.rectangle(img, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
            # XXX: hack to enable matching of generators
            boxes.append((X+5, Y+5, X-5 + W, Y-5 + H))

        return boxes

    def match_img_symbols(self, img_gray_cropped, symbols):
        """ matches an image against set of images """
        for k, v in symbols.items():
            rot1 = cv2.rotate(v, cv2.ROTATE_90_CLOCKWISE)
            rot2 = cv2.flip(rot1, 0)

            for template in [v, cv2.flip(v, 1), cv2.flip(v, 0), rot1, rot2]:
                # print('search {} = {}'.format(k, type(v)))
                # template = v

                w, h = template.shape[::-1]

                # iw, ih = img_gray_cropped.shape[::-1]
                # print(w,h,iw,ih)

                # Perform match operations.
                try:
                    res = cv2.matchTemplate(
                        img_gray_cropped, template, cv2.TM_CCOEFF_NORMED)
                except:
                    continue
                loc = np.where(res >= MATCHING_THRESHOLD)

                # Draw a rectangle around the matched region.
                points = []
                for pt in zip(*loc[::-1]):
                    return {k: [pt[0]+w/2, pt[1]+h/2]}

        return None

    def detect_symbol(self, img_rgb, bboxes, symbols_path='data/symbols_png/'):
        """ given image and bounding boxes, classifies symbols
            using template matching
        """

        # load symbol templates
        symbols = {}
        for sf in os.listdir(symbols_path):
            name = sf.split('.')[0]
            if name in ['Pipe', 'Terminal', 'Crossing']:
                continue
            symbol_grey = cv2.imread(symbols_path + sf, 0)
            if symbol_grey is None:
                print("ERROR: symbol image not found", sf)
                sys.exit(-1)
            symbols[name] = symbol_grey

        # test bounded image area against all symbols
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        nodes = []
        for bb in bboxes:
            img_gray_cropped = img_gray[bb[1]:bb[3], bb[0]:bb[2]]

            # TODO weird output
            node = self.match_img_symbols(img_gray_cropped, symbols)
            if node != None:
                node[list(node)[0]] = [int(bb[1] + (bb[3] - bb[1]) / 2.0),
                                       int(bb[0] + (bb[2] - bb[0]) / 2.0)]
                nodes.append(node)

        return nodes

    def build_graph(self, nodes, edges):
        """ connects lines and nodes to form a graph """
        #print("building graph")
        #print(nodes, edges)

        node_list = []
        edge_list = []

        id_counts = {}
        for node in nodes:
            node_type = list(node)[0]
            if node_type not in id_counts.keys():
                id_counts[node_type] = 0
            else:
                id_counts[node_type] += 1

            label = node_type
            if node_type in type_label_map:
                label = type_label_map[node_type]
            node_list.append(
                {'id': '{}{}'.format(node_type, id_counts[node_type]),
                 'label': label,
                 'coordinates': node[node_type]})

        for edge in edges:
            start = np.array([edge[1], edge[0]])
            end = np.array([edge[3], edge[2]])
            #print(start, end)
            best_source = None
            best_source_dist = np.Inf
            best_target = None
            best_target_dist = np.Inf
            for node in node_list:
                source_dist = np.linalg.norm(
                    start - np.array(node['coordinates']))
                target_dist = np.linalg.norm(
                    end - np.array(node['coordinates']))
                if best_source_dist > source_dist:
                    best_source = node['id']
                    best_source_dist = source_dist
                if best_target_dist > target_dist:
                    best_target = node['id']
                    best_target_dist = target_dist
            #print(best_source_dist, best_target_dist)
            edge_list.append({'source': best_source, 'target': best_target})

        graph = {'nodes': node_list, 'edges': edge_list}
        return graph

    def filter_and_merge_lines(self, lines):
        unique_lines = []
        # filter out duplicates
        for l in lines:
            if ((l[0]-l[2])**2 + (l[1]-l[3])**2) < DIST_THRESHOLD:
                #print('skipping short line', l)
                continue

            skip = False
            for ul in unique_lines:
                sd = (ul[0]-l[2])**2 + (ul[1]-l[3])**2
                ed = (ul[2]-l[0])**2 + (ul[3]-l[1])**2
                if sd < DIST_THRESHOLD and ed < DIST_THRESHOLD:
                    #print("duplicate")
                    skip = True
                    break
                sd = (ul[0]-l[0])**2 + (ul[1]-l[3])**2
                ed = (ul[2]-l[2])**2 + (ul[1]-l[3])**2
                if sd < DIST_THRESHOLD and ed < DIST_THRESHOLD:
                    #print("duplicate")
                    skip = True
                    break

            if skip:
                continue
            unique_lines.append(l)

        edges = []
        # merge
        for ul in unique_lines:
            skip = False
            for i, e in enumerate(edges):
                sd = (e[0]-ul[0])**2 + (e[1]-ul[1])**2
                ed = (e[2]-ul[2])**2 + (e[3]-ul[3])**2
                if ed < DIST_THRESHOLD:
                    edges[i] = [e[0], e[1], ul[0], ul[1]]
                    #print('merged line')
                    skip = True
                    break
                if sd < DIST_THRESHOLD:
                    edges[i] = [e[2], e[3], ul[2], ul[3]]
                    #print('merged line')
                    skip = True
                    break
                sd = (e[0]-ul[2])**2 + (e[1]-ul[3])**2
                ed = (e[2]-ul[0])**2 + (e[3]-ul[1])**2
                if ed < DIST_THRESHOLD:
                    edges[i] = [e[0], e[1], ul[2], ul[3]]
                    #print('merged line')
                    skip = True
                    break
                if sd < DIST_THRESHOLD:
                    edges[i] = [e[2], e[3], ul[0], ul[1]]
                    #print('merged line')
                    skip = True
                    break
            if not skip:
                edges.append(ul)

        #print(len(edges))
        # for line in edges:
        #     l = line
        #     # cv2.line(img_rgb, (int(l[0]), int(l[1])),
        #     #          (int(l[2]), int(l[3])), (0, 255, 0), 1, cv2.LINE_AA)

        # cv2.imshow('Image', img_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return edges

    def generate_graph_from_image(self, img):
        """ takes an image and generates a graph """
        print("...segmenting, filtering and merging lines...")
        lines = self.segment_lines(img)
        edges = self.filter_and_merge_lines(lines)

        print("...segmenting and detecting symbols...")
        bounding_boxes = self.segment_symbols(img)
        nodes = self.detect_symbol(img, bounding_boxes)

        print("...building graph...")
        graph = self.build_graph(nodes, edges)
        return graph


if __name__ == '__main__':
    graph_filename = 'data/examples_png/ex0.png'
    # graph_filename = r'data_SI/examples_png/ex0_hand.png'
    # graph_filename = r'data_SI/examples_png/ex0_pic.png'

    img_rgb = cv2.imread(graph_filename)
    if img_rgb is None:
        print("ERROR: image not found", graph_filename)
        sys.exit(-1)

    graph_generator = GraphGenerator(img_rgb)
    print(graph_generator.graph)
