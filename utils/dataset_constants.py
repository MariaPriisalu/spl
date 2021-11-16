class cityscapes_constants(object):


    def __init__(self):
        # Cities in different training sets
        self.train_extra = ['nuremberg', 'konigswinter', 'muhlheim-ruhr',
                            'schweinfurt', 'saarbrucken', 'mannheim', 'heidelberg',
                            'bad-honnef', 'dortmund', 'troisdorf', 'bayreuth',
                            'wurzburg', 'oberhausen', 'bamberg', 'dresden',
                            'augsburg', 'erlangen', 'freiburg', 'duisburg',
                            'karlsruhe', 'wuppertal', 'heilbronn', 'konstanz']

        self.train = ['darmstadt', 'tubingen', 'weimar', 'bremen', 'krefeld',
                      'ulm', 'jena', 'stuttgart', 'erfurt', 'strasbourg',
                      'cologne', 'zurich', 'hanover', 'hamburg', 'aachen',
                      'monchengladbach', 'bochum', 'dusseldorf']
        self.test = ['bielefeld', 'berlin', 'mainz', 'leverkusen', 'munich', 'bonn']
        self.val = ['frankfurt', 'munster', 'lindau']

        self.mapping_carla_to_cs = []
        self.mapping_carla_to_cs[0] = 0  # None
        self.mapping_carla_to_cs[1] = 11  # Building
        self.mapping_carla_to_cs[2] = 13  # Fence
        self.mapping_carla_to_cs[3] = 4  # Other/Static
        self.mapping_carla_to_cs[4] = 24  # Pedestrian
        self.mapping_carla_to_cs[5] = 17  # Pole
        self.mapping_carla_to_cs[6] = 7  # RoadLines
        self.mapping_carla_to_cs[7] = 7  # Road
        self.mapping_carla_to_cs[8] = 8  # Sidewlk
        self.mapping_carla_to_cs[9] = 21  # Vegetation
        self.mapping_carla_to_cs[10] = 26  # Vehicles
        self.mapping_carla_to_cs[11] = 12  # Wall
        self.mapping_carla_to_cs[12] = 20  # Traffic sign

        self.colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                   (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160),
                   (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153), (180, 165, 180),
                   (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                   (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                   (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
                   (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]

        self.labels = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground',
                  'road', 'sidewalk',
                  'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole',
                  'polegroup',
                  'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                  'bus',
                  'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

class carla_constants(cityscapes_constants):


    def __init__(self):

        self.mapping_carla_to_cs = []
        self.mapping_carla_to_cs[0] = 0  # None
        self.mapping_carla_to_cs[1] = 11  # Building
        self.mapping_carla_to_cs[2] = 13  # Fence
        self.mapping_carla_to_cs[3] = 4  # Other/Static
        self.mapping_carla_to_cs[4] = 24  # Pedestrian
        self.mapping_carla_to_cs[5] = 17  # Pole
        self.mapping_carla_to_cs[6] = 7  # RoadLines
        self.mapping_carla_to_cs[7] = 7  # Road
        self.mapping_carla_to_cs[8] = 8  # Sidewlk
        self.mapping_carla_to_cs[9] = 21  # Vegetation
        self.mapping_carla_to_cs[10] = 26  # Vehicles
        self.mapping_carla_to_cs[11] = 12  # Wall
        self.mapping_carla_to_cs[12] = 20  # Traffic sign


        self.train = list(range(0,100))
        self.test = list(range(100, 150))
        self.val = list(range(0, 250)) # On the validation set.

        self.labels_carla = ['unlabeled', 'building', 'fence', 'static', 'person', 'pole', 'roadlines', 'road',
                       'sidewalk', 'vegetation','vehicles', 'wall', 'traffic sign']