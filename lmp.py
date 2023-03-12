import openai
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = "code-davinci-002"

openai.api_key = openai_api_key

def lmp(prompt, stop_tokens=None, query_kwargs=None):
    use_query_kwargs = {
        'engine': model_name,
        'max_tokens': 1024,
        'temperature': 0,
    }
    if query_kwargs is not None:
      use_query_kwargs.update(query_kwargs)

    response = openai.Completion.create(
        prompt=prompt, stop=stop_tokens, **use_query_kwargs
    )['choices'][0]['text'].strip()

    print(prompt)
    print(response)

    return response

prompt_combined_parse_pos = '''
from utils import get_obj_pos

# right of the red block.
ret_val = get_obj_pos('red block') + np.array([0.3, 0])
# a bit below the cyan bowl.
ret_val = get_obj_pos('cyan bowl') + np.array([0, -0.1])
# a point between the cyan block and purple bowl.
pos = get_mid_point_np(pt0=get_obj_pos('cyan block'), pt1=get_obj_pos('purple bowl'))
ret_val = pos
# the corner closest to the yellow block.
corner_positions = get_corner_positions()
closest_corner_idx = get_closest_idx(points=corner_positions, point=get_obj_pos('yellow block'))
closest_corner_pos = corner_positions(closest_corner_idx)
ret_val = closest_corner_pos
'''.strip()

prompt_combined_parse_obj = '''
import numpy as np
from utils import get_obj_pos, get_objs_bigger_than_area_th

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the blocks.
ret_val = ['brown block', 'blue block']
# the sky colored block.
ret_val = 'blue block'
objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block']
# the right most block.
block_names = ['cyan block', 'pink block', 'blue block']
block_positions = np.array([get_obj_pos(block_name) for block_name in block_names])
right_block_name = block_names[np.argmax(block_positions[:, 0])]
ret_val = right_block_name
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# blocks above the brown bowl.
block_names = ['blue block', 'cyan block', 'purple block']
brown_bowl_pos = get_obj_pos('brown bowl')
use_block_names = [name for name in block_names if get_obj_pos(name)[1] > brown_bowl_pos[1]]
ret_val = use_block_names
'''.strip()

prompt_combined_ui = '''
import numpy as np
from utils import get_obj_pos, put_first_on_second, parse_obj

objects = ['green block', 'green bowl', 'yellow block', 'yellow bowl']
# move the bowls below the yellow block upwards.
bowl_names = parse_obj('the bowls below the yellow block', f'objects = {objects}')
for bowl_name in bowl_names:
  target_pos = get_obj_pos(bowl_name) + np.array([0, 0.1])
  put_first_on_second(bowl_name, target_pos)
objects = ['white block', 'white bowl', 'yellow block', 'yellow bowl']
# if the banana colored bowl is the top most bowl, then place it on the white bowl.
bowl_name = parse_obj('the banana colored bowl', f'objects = {objects}')
top_bowl_name = parse_obj('the top most bowl', f'objects = {objects}')
if bowl_name == top_bowl_name:
  put_first_on_second(bowl_name, 'white bowl')
objects = ['blue block', 'blue bowl', 'red block', 'red bowl']
# place the block closest to the blue bowl on the other bowl.
block_name = parse_obj('the block closest to the blue bowl', f'objects = {objects}')
bowl_name = parse_obj('a bowl other than the blue bowl', f'objects = {objects}')
put_first_on_second(block_name, 'red bowl')
'''.strip()

big_prompt = '''
import numpy as np
from utils import get_obj_pos, get_objs_bigger_than_area_th, put_first_on_second, parse_obj

objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the blocks.
ret_val = ['brown block', 'blue block']
# the sky colored block.
ret_val = 'blue block'
objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block']
# the right most block.
block_names = ['cyan block', 'pink block', 'blue block']
block_positions = np.array([get_obj_pos(block_name) for block_name in block_names])
right_block_name = block_names[np.argmax(block_positions[:, 0])]
ret_val = right_block_name
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# blocks above the brown bowl.
block_names = ['blue block', 'cyan block', 'purple block']
brown_bowl_pos = get_obj_pos('brown bowl')
use_block_names = [name for name in block_names if get_obj_pos(name)[1] > brown_bowl_pos[1]]
ret_val = use_block_names

objects = ['green block', 'green bowl', 'yellow block', 'yellow bowl']
# move the bowls below the yellow block upwards.
bowl_names = parse_obj('the bowls below the yellow block', f'objects = {objects}')
for bowl_name in bowl_names:
  target_pos = get_obj_pos(bowl_name) + np.array([0, 0.1])
  put_first_on_second(bowl_name, target_pos)
objects = ['white block', 'white bowl', 'yellow block', 'yellow bowl']
# if the banana colored bowl is the top most bowl, then place it on the white bowl.
bowl_name = parse_obj('the banana colored bowl', f'objects = {objects}')
top_bowl_name = parse_obj('the top most bowl', f'objects = {objects}')
if bowl_name == top_bowl_name:
  put_first_on_second(bowl_name, 'white bowl')
objects = ['blue block', 'blue bowl', 'red block', 'red bowl']
# place the block closest to the blue bowl on the other bowl.
block_name = parse_obj('the block closest to the blue bowl', f'objects = {objects}')
bowl_name = parse_obj('a bowl other than the blue bowl', f'objects = {objects}')
put_first_on_second(block_name, 'red bowl')
'''.strip()

bowl_on_block_interrupt = '''
objects = ['red block', 'blue bowl', 'blue block', 'red bowl']
# put the bowls on the blocks of matching color.                 
red_bowl_name = parse_obj('the red bowl', f'objects = {objects}')
blue_bowl_name = parse_obj('the blue bowl', f'objects = {objects}')
red_block_name = parse_obj('the red block', f'objects = {objects}') 
blue_block_name = parse_obj('the blue block', f'objects = {objects}')
original_positions = np.array([get_obj_pos(name) for name in names])
put_first_on_second(red_bowl_name, red_block_name)
query = "# return everything back to their original positions and put the blocks on the bowls"
'''.strip()

bowl_on_block_interrupt_hint = '''
objects = ['red block', 'blue bowl', 'blue block', 'red bowl']
# put the bowls on the blocks of matching color.                 
red_bowl_name = parse_obj('the red bowl', f'objects = {objects}')
blue_bowl_name = parse_obj('the blue bowl', f'objects = {objects}')
red_block_name = parse_obj('the red block', f'objects = {objects}') 
blue_block_name = parse_obj('the blue block', f'objects = {objects}')
names = [red_bowl_name, blue_bowl_name, red_block_name, blue_block_name]
original_positions = np.array([get_obj_pos(name) for name in names])
put_first_on_second(red_bowl_name, red_block_name)
# return everything back to their original positions and put the blocks on the bowls
for name, original_position in zip(names, original_positions):
    put_first_on_second(name, original_position)
'''.strip()

interrupt_prompt = '''
objects = ['red block', 'blue bowl', 'blue block', 'red bowl']
# put the bowls on the blocks of matching color.                 
red_bowl_name = parse_obj('the red bowl', f'objects = {objects}')
blue_bowl_name = parse_obj('the blue bowl', f'objects = {objects}')
red_block_name = parse_obj('the red block', f'objects = {objects}') 
blue_block_name = parse_obj('the blue block', f'objects = {objects}')
names = [red_bowl_name, blue_bowl_name, red_block_name, blue_block_name]
original_positions = np.array([get_obj_pos(name) for name in names])
put_first_on_second(red_bowl_name, red_block_name)
# return everything back to their original positions and put the blocks on the bowls
for name, original_position in zip(names, original_positions):
    put_first_on_second(name, original_position)
put_first_on_second(red_block_name, red_bowl_name)
put_first_on_second(blue_block_name, blue_bowl_name)
'''.strip()

bowl_on_block_interrupt_color = '''
objects = ['red block', 'blue bowl', 'blue block', 'red bowl']
# put the blocks on the bowls of matching color.
red_bowl_name = parse_obj('the red bowl', f'objects = {objects}')
blue_bowl_name = parse_obj('the blue bowl', f'objects = {objects}')
red_block_name = parse_obj('the red block', f'objects = {objects}') 
blue_block_name = parse_obj('the blue block', f'objects = {objects}')
original_positions = np.array([get_obj_pos(name) for name in names])
put_first_on_second(red_bowl_name, red_block_name)
# put things back and put the same color objects together
'''.strip()

bowl_on_block_interrupt_shape = '''
objects = ['red block', 'blue bowl', 'blue block', 'red bowl']
# put the blocks on the bowls of matching color.
red_bowl_name = parse_obj('the red bowl', f'objects = {objects}')
blue_bowl_name = parse_obj('the blue bowl', f'objects = {objects}')
red_block_name = parse_obj('the red block', f'objects = {objects}') 
blue_block_name = parse_obj('the blue block', f'objects = {objects}')
original_positions = np.array([get_obj_pos(name) for name in names])
put_first_on_second(red_bowl_name, red_block_name)
# put things back and put the same shape objects together
'''.strip()

cleanup_interrupt_prompt_undo = '''
objects = ["red block", "red bin", "red bowl", "blue block", "blue bin", "blue bowl", "yellow block", "yellow bin", "yellow bowl"]
# put the blocks away
red_bowl_name = parse_obj('the red bowl', f'objects = {objects}')
red_block_name = parse_obj('the red block', f'objects = {objects}') 
red_bin_name = parse_obj('the red bin', f'objects = {objects}') 
blue_bowl_name = parse_obj('the blue bowl', f'objects = {objects}')
blue_block_name = parse_obj('the blue block', f'objects = {objects}') 
blue_bin_name = parse_obj('the blue bin', f'objects = {objects}') 
yellow_bowl_name = parse_obj('the yellow bowl', f'objects = {objects}')
yellow_block_name = parse_obj('the yellow block', f'objects = {objects}') 
yellow_bin_name = parse_obj('the yellow bin', f'objects = {objects}') 
red_bowl_position = get_obj_pos(red_bowl_name)
red_block_position = get_obj_pos(red_block_name)
red_bin_position = get_obj_pos(red_bin_name)
blue_bowl_position = get_obj_pos(blue_bowl_name)
blue_block_position = get_obj_pos(blue_block_name)
blue_bin_position = get_obj_pos(blue_bin_name)
yellow_bowl_position = get_obj_pos(yellow_bowl_name)
yellow_block_position = get_obj_pos(yellow_block_name)
yellow_bin_position = get_obj_pos(yellow_bin_name)
put_first_on_second(red_block_name, red_bin_name)
put_first_on_second(blue_block_name, blue_bin_name)
put_first_on_second(yellow_block_name, yellow_bin_name)
# actually do not put anything yellow away. put the yellow block back because of this.
put_first_on_second(yellow_block_name, yellow_block_position)
# now put the bowls away
put_first_on_second(red_bowl_name, red_bin_name)
put_first_on_second(blue_bowl_name, blue_bin_name)
objects = ["red spoon", "red bin", "red fork", "blue spoon", "blue bin", "blue fork", "yellow spoon", "yellow bin", "yellow fork"]
# now put the spoons away
'''.strip()


"""
# small sanity check for parsing
context = "objects = ['red block', 'blue bowl', 'blue block', 'red bowl']"
query = '# put the bowls on the blocks of matching color.'
print(context)
_ = lmp(f'{prompt_combined_ui}\n{context}', query, ['#', 'objects = ['])
print()
"""

# Experiment 1: single preference, single block
"""
# example with interrupt but no example
context = "objects = ['red block', 'blue bowl', 'blue block', 'red bowl']"
prompt = f'{big_prompt}\n{bowl_on_block_interrupt}'
print(prompt)
_ = lmp(prompt, ['#', 'objects = ['])
print()
"""

"""
# example with interrupt and hint
prompt = f'{big_prompt}\n{bowl_on_block_interrupt_hint}'
#print(prompt)
_ = lmp(prompt, ['#', 'objects = ['])
print()
"""

"""
# example with interrupt in prompt
prompt = f'{big_prompt}\n{interrupt_prompt}\n{bowl_on_block_interrupt_color}'
_ = lmp(prompt, ['#', 'objects = ['])
print()
"""

"""
# example with interrupt in prompt, different query
prompt = f'{big_prompt}\n{interrupt_prompt}\n{bowl_on_block_interrupt_shape}'
print(prompt)
_ = lmp(prompt, ['#', 'objects = ['])
print()
# put things back and group by shape doesnt work. cant generalize color to shape
"""

# Experiment 2: single preference, multiple blocks
big_interrupt_prompt = f"{big_prompt}\n{interrupt_prompt}"
prompt = f"{big_interrupt_prompt}\n{cleanup_interrupt_prompt_undo}"
_ = lmp(prompt, ['#', 'objects = ['])
print()

# Experiment 3: multiple preference, multiple blocks
 
# Experiment 4: conflicting preferences, multiple blocks
