import numpy as np
import vtk
from Misc.Utils import FindPolarAngle,EvaluatePoint

"""!@package 
This package contains a number of vtk functions that helps to render the geometry of an experimental setup
"""


"""!@brief
Default colors:
    Background => white 
    Text => Black
    TR => Blue
PET:
    FanSize=> Green
    Crystals => Red
CT: 
    Source => Red
    Detector => Green
"""
par_pixel_color = (0, 1, 0)
par_background_color = (1, 1, 1)
par_text_color = (0, 0, 0)
par_tr_color = (0, 0, 1)
par_src_color = (1, 0, 0)
par_pixel_color = (0, 1, 0)


def CreateAxesActor(fontsize=10, axis_lenght=30):
    """!@brief 
        Generate the Cartesian axis and return the relative actor 
    """
    axes = vtk.vtkAxesActor()
    txtprop = vtk.vtkTextProperty()
    txtprop.SetFontSize(fontsize)
    txtprop.ShadowOff()
    txtprop.SetColor(par_text_color)
    axes.GetXAxisTipProperty().SetColor(par_text_color)
    axes.SetXAxisLabelText("x")
    axes.GetXAxisShaftProperty().SetColor(par_text_color)
    axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
    axes.SetYAxisLabelText("y")
    axes.GetYAxisTipProperty().SetColor(par_text_color)
    axes.GetYAxisShaftProperty().SetColor(par_text_color)
    axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
    axes.SetZAxisLabelText("z")
    axes.GetZAxisTipProperty().SetColor(par_text_color)
    axes.GetZAxisShaftProperty().SetColor(par_text_color)
    axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(txtprop)
    axes.SetTotalLength(axis_lenght, axis_lenght, axis_lenght)
    return axes


def CreateTR(TRSizemm, VoxelSizemm, color=par_tr_color, edge_color=par_text_color):
    """!@brief
     Create the cube grid representing the TR and  return the relative actor 
    """
    TR_grid = vtk.vtkStructuredGrid()
    #
    Points = vtk.vtkPoints()
    Nx = np.rint(TRSizemm[0] / VoxelSizemm[0]).astype(int) + 1
    Ny = np.rint(TRSizemm[1] / VoxelSizemm[1]).astype(int) + 1
    Nz = np.rint(TRSizemm[2] / VoxelSizemm[2]).astype(int) + 1
    offset_x = -TRSizemm[0] / 2.0
    offset_y = -TRSizemm[1] / 2.0
    offset_z = -TRSizemm[2] / 2.0
    for _z in range(Nz):
        for _y in range(Ny):
            for _x in range(Nx):
                Points.InsertNextPoint(
                    offset_x + _x * VoxelSizemm[0],
                    offset_y + _y * VoxelSizemm[1],
                    offset_z + _z * VoxelSizemm[2],
                )
    TR_grid.SetDimensions(Nx, Ny, Nz)
    TR_grid.SetPoints(Points)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(TR_grid)
    TRActor = vtk.vtkActor()
    TRActor.SetMapper(mapper)
    TRActor.GetProperty().EdgeVisibilityOn()
    TRActor.GetProperty().SetEdgeColor(edge_color)
    TRActor.GetProperty().SetColor(color)
    TRActor.GetProperty().SetOpacity(0.1)
    return TRActor


def CreateLabel(point, text):
    """!@brief Create a text label  and return the actor
    """
    atext = vtk.vtkBillboardTextActor3D()
    atext.SetInput(text)
    atext.SetPosition(point)
    atext.GetTextProperty().SetColor(par_text_color)
    return atext


def CreateCornerAnnotation(window_pos_2d, text, font_size=14):
    """!@brief 
    Create  2D text and put it in a certain position 
    """

    textActor = vtk.vtkTextActor()
    textActor.SetInput(text)
    textActor.SetPosition2(window_pos_2d)
    textActor.GetTextProperty().SetFontSize(font_size)
    textActor.GetTextProperty().SetColor(par_text_color)
    return textActor

def CreateAutoOrientedeCube(center, size, color):
    """!@brief
        Create a cube facing the center of the TR
    """
    # create the cube_src representing the crystal
    cube_src = vtk.vtkCubeSource()
    # depth
    cube_src.SetXLength(size[0])
    # trans-axial
    cube_src.SetYLength(size[2])
    # axial
    cube_src.SetZLength(size[1])
    # set the center of the cube_src in the TR center
    new_center = np.array(center.tolist())
    orientation=np.copy(new_center)
    orientation[2]=0
    radius_mm=np.sqrt(new_center[0]**2+new_center[1]**2)
    angle=FindPolarAngle(orientation,360.0)
    new_center = EvaluatePoint(new_center, orientation, size[2])
    cube_src.SetCenter([0,radius_mm,new_center[2]])
    cube_src.Update()
    # mapper
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputData(cube_src.GetOutput())
    # actor
    cubeActor = vtk.vtkActor()
    #
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetColor(color)
    cubeActor.GetProperty().SetEdgeColor(par_text_color)
    cubeActor.RotateZ(angle)
    return cubeActor

def CreateCube(center, size, angle, orientation, color):
    """!@brief
        Create a cube and return the actor
    """
    # create the cube_src representing the crystal
    cube_src = vtk.vtkCubeSource()
    # depth
    cube_src.SetXLength(size[0])
    # trans-axial
    cube_src.SetYLength(size[2])
    # axial
    cube_src.SetZLength(size[1])
    # set the center of the cube_src in the TR center
    new_center = np.array(center.tolist())
    new_center = EvaluatePoint(new_center, orientation, size[2])
    cube_src.SetCenter(new_center)
    cube_src.Update()
    # mapper
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputData(cube_src.GetOutput())
    # actor
    cubeActor = vtk.vtkActor()
    #
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetColor(color)
    cubeActor.GetProperty().SetEdgeColor(par_text_color)
    cubeActor.RotateZ(angle)
    return cubeActor


def CreateSphere(pos, radius_mm, color):
    """!@brief
        Create a sphere and return the actor
    """
    sphere_src = vtk.vtkSphereSource()
    sphere_src.SetCenter(pos)
    sphere_src.SetRadius(radius_mm)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere_src.GetOutputPort())
    shpere_actor = vtk.vtkActor()
    shpere_actor.SetMapper(mapper)
    shpere_actor.GetProperty().SetColor(color)
    shpere_actor.GetProperty().SetOpacity(1)
    return shpere_actor


def RenderSceneJupyter(renderer, w=800, h=800):
    """!@brief
        Takes vtkRenderer instance and returns an IPython Image with the rendering.
        This code was taken https://nbviewer.jupyter.org/gist/certik/5723420 
    """
    from IPython.display import Image

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(w, h)
    renderWindow.Render()
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = memoryview(writer.GetResult()).tobytes()
    return Image(data)


def CreateLine(p0, p1,width,color):
    '''!@brief
        Draw a line connecting two points and return the actor
    '''    
    line_src = vtk.vtkLineSource()
    line_src.SetPoint1(p0)
    line_src.SetPoint2(p1)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line_src.GetOutputPort())
    line_actor = vtk.vtkActor()
    line_actor.SetMapper(mapper)
    line_actor.GetProperty().SetLineWidth(width)
    line_actor.GetProperty().SetColor(color)
    return line_actor

