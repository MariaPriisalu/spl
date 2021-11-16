from RealTimeCarlaUtils import *
from CarlaServerConnection import *
from CarlaEnvironmentActors import  *
import carla

import no_rendering_mode as SIMPLIFIED_RENDERING

class EnvironmentRendering():
    def __init__(self, renderingOptions : RenderOptions, dataGatheringParam : DataGatherParams, args):
        self.renderingOptions = renderingOptions
        self.RenderImgType = None
        self.font = None
        self.clock = None
        self.display = None
        self.dataGatheringParam = dataGatheringParam

        # HACKS FOR RENDERING
        args.map = None
        args.show_triggers = False
        args.show_connections = False
        args.show_spawn_points = False
        args.filter = "vehicle.*"
        args.use_hero_actor = args.heroPerspectiveView

        if renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            pygame.init()
            #UsePyGameRendering()
            if renderingOptions.sceneRenderType == RenderType.RENDER_COLORED:
                self.RenderImgType = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
                self.display = pygame.display.set_mode((self.dataGatheringParam.image_size[0], self.dataGatheringParam.image_size[1]),
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self.RenderImgType = RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW
                self.display = pygame.display.set_mode((self.dataGatheringParam.image_size[0], self.dataGatheringParam.image_size[1]),
                                                       pygame.HWSURFACE | pygame.DOUBLEBUF)


            self.font = RenderUtils.get_font()
            self.clock = pygame.time.Clock()

            self.input_control = SIMPLIFIED_RENDERING.InputControl("test")
            self.hud = SIMPLIFIED_RENDERING.HUD("hud", args.width, args.height)
            self.world = SIMPLIFIED_RENDERING.World("world", args, timeout=2.0)
            # For each module, assign other modules that are going to be used inside that module
            self.input_control.start(self.hud, self.world)
            self.hud.start()
            self.world.start(self.hud, self.input_control)

    def setFixedObserverPos(self, location):
        if self.renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            self.world.fixedObserverPos = location

    def tick(self, syncData):
        if self.renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            self.clock.tick()

            # TODO: fix this call to get in input control
            self.processInputEvents()

            # Tick all modules
            self.world.tick(self.clock)
            self.hud.tick(self.clock)
            self.input_control.tick(self.clock)

            worldSnapshot = syncData['worldSnapshot']

            # TODO: fix the hero camera viewport !!!!
            """
            if self.renderingOptions.sceneRenderType == RenderType.RENDER_COLORED:
                # Take the date from world considering hero car view
                image_seg = syncData["seg"]
                image_rgb = syncData["rgb"]
                image_depth = syncData["depth"]

                # Draw the display and stats
                if self.RenderImgType ==  RenderUtils.EventType.EV_SWITCH_TO_DEPTH:
                    RenderUtils.draw_image(self.display, image_depth, blend=True)
                elif self.RenderImgType == RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG:
                    RenderUtils.draw_image(self.display, image_rgb)
                    RenderUtils.draw_image(self.display, image_seg, blend=True)
            else:
                # TODO
                pass

            # Hud rendering
            self.display.blit(self.font.render('%Press D - Depth or S - Segmentation + RGB or T for topview', True, (255, 255, 255)), (8, 5))
            self.display.blit(self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)), (8, 20))

            fps = round(1.0 / worldSnapshot.timestamp.delta_seconds)
            self.display.blit(self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),(8, 38))
            """

            # Render modules
            self.display.fill(SIMPLIFIED_RENDERING.COLOR_ALUMINIUM_4)
            self.world.render(self.display)
            self.hud.render(self.display)
            #input_control.render(display)

            pygame.display.flip()

    def quit(self):
        if self.renderingOptions.sceneRenderType != RenderType.RENDER_NONE:
            pygame.quit()

    def processInputEvents(self):



        # TODO: fix this
        """
        inputEv = RenderUtils.get_input_event()
        if inputEv == RenderUtils.EventType.EV_QUIT:
            return False
        elif inputEv == RenderUtils.EventType.EV_SWITCH_TO_DEPTH:
            self.RenderImgType = RenderUtils.EventType.EV_SWITCH_TO_DEPTH
        elif inputEv == RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG:
            self.RenderImgType = RenderUtils.EventType.EV_SWITCH_TO_RGBANDSEG
        elif inputEv == RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW:
            self.RenderImgType = RenderUtils.EventType.EV_SIMPLIFIED_TOPVIEW
        """