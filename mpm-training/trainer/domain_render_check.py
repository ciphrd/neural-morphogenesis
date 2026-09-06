"""Offscreen GPU checks for true triangle outlines, zoom and periodic seams.

Run: .venv/bin/python domain_render_check.py [optional-preview.png]
"""
from pathlib import Path
import sys
import numpy as np
import wgpu
from triangle_vertices import vertices_from_edges
from device import pick_device
from shader_template import template_shader


def check():
    device = pick_device()
    source = (Path(__file__).resolve().parents[1]/'viewer/src/gpu/render.wgsl').read_text()
    module = device.create_shader_module(code=template_shader(source, {'CHANNELS': 8}))
    pipeline = device.create_render_pipeline(
        layout='auto', vertex={'module': module, 'entry_point': 'domainVertex'},
        primitive={'topology': 'line-list'},
        fragment={'module': module, 'entry_point': 'domainFragment',
                  'targets': [{'format': 'rgba8unorm'}]})
    size = 512
    texture = device.create_texture(size=(size,size,1),format='rgba8unorm',
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    images = []
    for center, edges, zoom in [([.5,.5],[[.3,.08],[.02,.25]],1),
                                 ([.005,.005],[[.3,.08],[.02,.25]],1),
                                 ([.5,.5],[[.09,.02],[.01,.07]],4)]:
        center, edges = np.array(center,np.float32), np.array(edges,np.float32)
        rest = np.zeros(20,np.float32)
        rest[12:18] = vertices_from_edges([center],[edges])[0]
        # Zero marker alpha and appearance: outlines must remain visible.
        positions = device.create_buffer_with_data(data=center,usage=wgpu.BufferUsage.STORAGE)
        domains = device.create_buffer_with_data(data=rest,usage=wgpu.BufferUsage.STORAGE)
        view = device.create_buffer_with_data(data=np.array([zoom,0,0,0],np.float32),usage=wgpu.BufferUsage.UNIFORM)
        geometry = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[
            {'binding':4,'resource':{'buffer':domains}}])
        camera = device.create_bind_group(layout=pipeline.get_bind_group_layout(1),entries=[
            {'binding':0,'resource':{'buffer':view}}])
        encoder = device.create_command_encoder()
        render = encoder.begin_render_pass(color_attachments=[{
            'view':texture.create_view(),'load_op':'clear','store_op':'store','clear_value':(0,0,0,1)}])
        render.set_pipeline(pipeline)
        render.set_bind_group(0,geometry)
        render.set_bind_group(1,camera)
        render.draw(54,1)
        render.end()
        device.queue.submit([encoder.finish()])
        raw = device.queue.read_texture({'texture':texture}, {'bytes_per_row':size*4}, (size,size,1))
        rgba = np.frombuffer(raw,np.uint8).reshape(size,size,4)
        mask = rgba[:,:,2]>0
        yy,xx = np.nonzero(mask)
        assert len(xx)>100, 'Missing triangle outlines'
        pixels = np.column_stack((xx+.5,yy+.5))
        a = center-edges.sum(axis=1)/3
        triangle = np.array([a,a+edges[:,0],a+edges[:,1]])
        distance = np.full(len(pixels),np.inf)
        for tx in (-1,0,1):
            for ty in (-1,0,1):
                screen = ((triangle+[tx,ty]-.5)*zoom+.5)*size
                screen[:,1] = size-screen[:,1]
                for start,end in zip(screen,np.roll(screen,-1,axis=0)):
                    edge = end-start
                    t = np.clip((pixels-start)@edge/(edge@edge),0,1)
                    distance = np.minimum(distance,np.linalg.norm(pixels-start-t[:,None]*edge,axis=1))
                    midpoint = (start+end)/2
                    if np.all(midpoint>3) and np.all(midpoint<size-3):
                        x,y = midpoint.astype(int)
                        assert mask[y-2:y+3,x-2:x+3].any(), 'Missing triangle edge'
        assert distance.max()<1.5, ('Outline disagrees with stored geometry',distance.max())
        if center[0]<.01:
            assert not mask[128:384,128:384].any(), 'Wrapped vertices created a false seam-spanning edge'
            assert mask[:80].any() and mask[-80:].any() and mask[:,:80].any() and mask[:,-80:].any()
        images.append(rgba.copy())
        positions.destroy(); domains.destroy(); view.destroy()
    texture.destroy()
    if len(sys.argv)>1:
        from PIL import Image
        Image.fromarray(np.concatenate(images,axis=1)).save(sys.argv[1])
    print('[PASS] rendered outlines match domain edges at 1x/4x zoom, wrap both seams, and ignore marker opacity')


if __name__ == '__main__':
    check()
