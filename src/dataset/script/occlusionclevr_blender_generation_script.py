"""Blender-side rendering script for generating Occlusion CLEVR images from explicit object specs.

IMPORTANT:
- This script is intended to be run *inside Blender*:
    blender --background --python occlusionclevr_blender_generation_script.py -- --help
- It relies on facebookresearch/clevr-dataset-gen's image_generation/utils.py and assets.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def _parse_args(argv: List[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--clevr_gen_dir", required=True, help="Path to the clevr-dataset-gen repository root")
	parser.add_argument("--spec_jsonl", required=True, help="Path to a JSONL file; each line is a scene spec")
	parser.add_argument("--output_image_dir", required=True, help="Directory to write PNG images")
	parser.add_argument("--output_scene_dir", default=None, help="Optional directory to write per-image scene JSON")
	parser.add_argument("--width", type=int, default=224)
	parser.add_argument("--height", type=int, default=224)
	parser.add_argument("--render_num_samples", type=int, default=64)
	parser.add_argument("--use_gpu", type=int, default=0)
	parser.add_argument(
		"--gpu_backend",
		default="CUDA",
		help="Cycles backend (only CUDA supported in this project)",
	)
	parser.add_argument(
		"--gpu_device_index",
		type=int,
		default=None,
		help=(
			"If set and --use_gpu=1, force Cycles to use only this CUDA GPU index "
			"(0-based among CUDA devices visible to Blender)"
		),
	)
	parser.add_argument(
		"--gpu_device_name",
		default=None,
		help=(
			"If set and --use_gpu=1, force Cycles to use only the first CUDA GPU whose name "
			"contains this substring (case-insensitive). Overrides --gpu_device_index."
		),
	)
	parser.add_argument("--min_dist", type=float, default=0.25)
	parser.add_argument("--margin", type=float, default=0.4)
	parser.add_argument("--max_retries", type=int, default=50)
	parser.add_argument("--key_light_jitter", type=float, default=1.0)
	parser.add_argument("--fill_light_jitter", type=float, default=1.0)
	parser.add_argument("--back_light_jitter", type=float, default=1.0)
	parser.add_argument("--camera_jitter", type=float, default=0.5)
	parser.add_argument(
		"--camera_elevation",
		type=float,
		default=None,
		help=(
			"If set, overrides the Camera Z location (world units) before per-image jitter. "
			"Lower values tend to increase occlusion; higher values tend to reduce occlusion."
		),
	)
	parser.add_argument(
		"--camera_pitch_deg",
		type=float,
		default=None,
		help=(
			"If set, overrides the Camera pitch angle (degrees) as rotation_euler.x before per-image jitter. "
			"More negative/less downward pitch tends to increase occlusion; more downward pitch tends to reduce it."
		),
	)
	return parser.parse_args(argv)


def _extract_blender_argv() -> List[str]:
	"""Extract argv items after '--' when invoked by Blender."""
	if "--" not in sys.argv:
		return []
	idx = sys.argv.index("--")
	return sys.argv[idx + 1 :]


def main() -> None:
	argv = _extract_blender_argv()
	args = _parse_args(argv)

	# Import Blender modules (only available inside Blender).
	import bpy  # type: ignore
	from mathutils import Vector  # type: ignore

	# Ensure we can import clevr-dataset-gen image_generation utils.
	image_gen_dir = os.path.join(args.clevr_gen_dir, "image_generation")
	if not os.path.isdir(image_gen_dir):
		raise FileNotFoundError(f"Expected image_generation/ under clevr_gen_dir: {image_gen_dir}")
	if image_gen_dir not in sys.path:
		sys.path.insert(0, image_gen_dir)

	import utils  # type: ignore

	# ----------------------------
	# Blender 5.x compatibility layer
	# ----------------------------
	# The upstream clevr-dataset-gen code targets Blender 2.7x and uses legacy
	# bpy selection APIs and the older bpy.ops.wm.append signature.
	# Implement the few operations we need using Blender 2.8+ / 5.x APIs.
	def _append_from_library(*, directory: str, name: str) -> None:
		directory = os.path.abspath(directory)
		if not directory.endswith(os.sep):
			directory += os.sep
		# IMPORTANT (Blender 5.0): filepath=... does NOT work for pseudo-paths like
		#   /path/to/Mat.blend/NodeTree/Mat
		# but directory+filename does.
		try:
			bpy.ops.wm.append(directory=directory, filename=name)
		except Exception:
			filepath = os.path.join(directory, name)
			bpy.ops.wm.append(filepath=filepath)

	def delete_object(obj) -> None:
		bpy.ops.object.select_all(action="DESELECT")
		obj.select_set(True)
		bpy.context.view_layer.objects.active = obj
		bpy.ops.object.delete()

	def load_materials(material_dir: str) -> None:
		for fn in os.listdir(material_dir):
			if not fn.endswith(".blend"):
				continue
			name = os.path.splitext(fn)[0]
			# Append the node group from: <file>.blend/NodeTree/<name>
			directory = os.path.join(material_dir, fn, "NodeTree")
			_append_from_library(directory=directory, name=name)

	def add_object(object_dir: str, name: str, scale: float, loc, theta: float = 0.0):
		# First figure out how many of this object already exist, to build a unique name.
		count = 0
		for obj in bpy.data.objects:
			if obj.name.startswith(name):
				count += 1

		# Load object from: <name>.blend/Object/<name>
		directory = os.path.join(object_dir, f"{name}.blend", "Object")
		_append_from_library(directory=directory, name=name)

		# Give it a unique name to avoid conflicts.
		new_name = f"{name}_{count}"
		bpy.data.objects[name].name = new_name
		obj = bpy.data.objects[new_name]

		# Set active and apply transforms without operators (more robust in headless).
		x, y = loc
		bpy.context.view_layer.objects.active = obj
		obj.rotation_euler[2] = theta
		obj.scale = (scale, scale, scale)
		obj.location = (x, y, scale)
		return obj

	data_dir = os.path.join(image_gen_dir, "data")
	base_scene_blendfile = os.path.join(data_dir, "base_scene.blend")
	properties_json = os.path.join(data_dir, "properties.json")
	shape_dir = os.path.join(data_dir, "shapes")
	material_dir = os.path.join(data_dir, "materials")

	if not os.path.exists(base_scene_blendfile):
		raise FileNotFoundError(f"Missing base scene: {base_scene_blendfile}")
	if not os.path.exists(properties_json):
		raise FileNotFoundError(f"Missing properties.json: {properties_json}")

	with open(properties_json, "r") as f:
		properties = json.load(f)

	color_name_to_rgba = {
		name: [float(c) / 255.0 for c in rgb] + [1.0]
		for name, rgb in properties["colors"].items()
	}
	# Map human-readable shape/material name -> blend resource name
	shape_name_to_blend = dict(properties["shapes"])  # e.g. cube -> SmoothCube_v2
	material_name_to_blend = dict(properties["materials"])  # e.g. rubber -> Rubber
	size_name_to_scale = dict(properties["sizes"])  # e.g. small -> 0.7

	os.makedirs(args.output_image_dir, exist_ok=True)
	if args.output_scene_dir is not None:
		os.makedirs(args.output_scene_dir, exist_ok=True)

	def _try_enable_gpu() -> None:
		"""Best-effort Cycles GPU enablement for Blender 5.x.

		Just setting scene.cycles.device = 'GPU' is often not enough; we also need to
		enable devices in preferences.
		"""
		try:
			cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
		except Exception:
			print("[warn] Cycles addon preferences not found; falling back to CPU")
			return

		backend = str(getattr(args, "gpu_backend", "CUDA") or "CUDA").upper()
		if backend != "CUDA":
			print(f"[warn] gpu_backend={backend} requested, but only CUDA is supported; forcing CUDA")
			backend = "CUDA"
		try:
			cycles_prefs.compute_device_type = backend
			cycles_prefs.get_devices()
			devs = list(getattr(cycles_prefs, "devices", []))
			if not devs:
				print("[warn] No Cycles devices found; falling back to CPU")
				return
			# Select exactly one CUDA device to avoid multi-GPU rendering overhead.
			cuda_devs = [d for d in devs if getattr(d, "type", "") == "CUDA"]
			cuda_names = [getattr(d, "name", "?") for d in cuda_devs]
			if not cuda_devs:
				print("[warn] Cycles compute_device_type=CUDA but no CUDA devices found; falling back to CPU")
				return

			chosen = None
			name_sub = getattr(args, "gpu_device_name", None)
			if name_sub:
				name_sub_l = str(name_sub).lower()
				for d in cuda_devs:
					if name_sub_l in str(getattr(d, "name", "")).lower():
						chosen = d
						break
				if chosen is None:
					print(f"[warn] No CUDA device name matched {name_sub!r}; available={cuda_names}; falling back to index")

			if chosen is None:
				idx = getattr(args, "gpu_device_index", None)
				if idx is None:
					idx = 0
				try:
					idx_int = int(idx)
				except Exception:
					idx_int = 0
				if idx_int < 0 or idx_int >= len(cuda_devs):
					print(
						f"[warn] gpu_device_index={idx_int} out of range for CUDA devices={cuda_names}; using 0"
					)
					idx_int = 0
				chosen = cuda_devs[idx_int]

			# Disable everything except the chosen CUDA device.
			for d in devs:
				try:
					d.use = (d == chosen)
				except Exception:
					pass

			print(
				"[info] Cycles GPU backend=CUDA selected="
				+ str(getattr(chosen, "name", "?"))
				+ " all_cuda="
				+ str(cuda_names)
			)
		except Exception as e:
			print(f"[warn] Failed to enable CUDA devices for Cycles: {e}; falling back to CPU")
			return

		try:
			bpy.context.scene.cycles.device = "GPU"
		except Exception:
			pass

	# ----------------------------
	# Batch rendering within one Blender process
	# ----------------------------
	# Opening the base .blend per-image is very slow. Instead:
	# - open base scene once
	# - load materials once
	# - for each spec: reset camera/lights, delete previous objects/materials, place, render
	bpy.ops.wm.open_mainfile(filepath=base_scene_blendfile)
	load_materials(material_dir)
	# Helps performance when rendering many frames/variants.
	try:
		bpy.context.scene.render.use_persistent_data = True
	except Exception:
		pass

	# render settings (constant)
	render_args = bpy.context.scene.render
	render_args.engine = "CYCLES"
	render_args.resolution_x = int(args.width)
	render_args.resolution_y = int(args.height)
	render_args.resolution_percentage = 100
	bpy.context.scene.cycles.samples = int(args.render_num_samples)

	# Low-angle / close-range camera settings can cause near-clip artifacts where geometry
	# disappears and the world background shows through as black. Make this more robust.
	try:
		cam_obj = bpy.data.objects["Camera"]
		if hasattr(cam_obj, "data") and hasattr(cam_obj.data, "clip_start"):
			cam_obj.data.clip_start = min(float(cam_obj.data.clip_start), 0.01)
			cam_obj.data.clip_end = max(float(cam_obj.data.clip_end), 500.0)
	except Exception:
		pass
	# Also ensure the world background is not pure black (helps when the camera sees outside
	# the backdrop at very shallow pitch angles).
	# try:
	# 	world = bpy.context.scene.world
	# 	if world is not None:
	# 		world.use_nodes = True
	# 		nodes = world.node_tree.nodes
	# 		bg = nodes.get("Background")
	# 		if bg is not None:
	# 			bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
	# 			bg.inputs[1].default_value = 1.0
	# except Exception:
	# 	pass

	if int(args.use_gpu) == 1:
		_try_enable_gpu()

	# Snapshot original light/camera locations so we can reset each image
	_cam0 = bpy.data.objects["Camera"].location.copy()
	_cam_rot0 = bpy.data.objects["Camera"].rotation_euler.copy()
	_lk0 = bpy.data.objects["Lamp_Key"].location.copy()
	_lf0 = bpy.data.objects["Lamp_Fill"].location.copy()
	_lb0 = bpy.data.objects["Lamp_Back"].location.copy()

	# Track created objects so we can delete between renders
	_created_objects = []

	# Cache materials so we don't recreate shaders every object/image.
	_material_cache: Dict[Tuple[str, Tuple[float, float, float, float]], Any] = {}

	def _get_or_create_material(mat_group_name: str, rgba: List[float]):
		key = (mat_group_name, (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])))
		if key in _material_cache:
			return _material_cache[key]

		mat = bpy.data.materials.new(name=f"MAT_{mat_group_name}_{len(_material_cache)}")
		mat.use_nodes = True
		nodes = mat.node_tree.nodes
		links = mat.node_tree.links
		for n in list(nodes):
			nodes.remove(n)

		out = nodes.new(type="ShaderNodeOutputMaterial")
		out.location = (400, 0)
		group = nodes.new(type="ShaderNodeGroup")
		group.location = (0, 0)
		group.node_tree = bpy.data.node_groups[mat_group_name]
		# Set the Color input if present
		for inp in group.inputs:
			if inp.name == "Color":
				inp.default_value = key[1]
				break
		links.new(group.outputs["Shader"], out.inputs["Surface"])

		_material_cache[key] = mat
		return mat

	def _rand(L: float) -> float:
		return 2.0 * L * (random.random() - 0.5)

	def _jitter_lights_and_camera() -> None:
		if args.camera_jitter > 0:
			cam = bpy.data.objects["Camera"]
			# If the user pins camera elevation, do not jitter Z; otherwise shallow-angle
			# views can randomly drop below/into the floor and produce dark bands.
			jitter_dims = [0, 1] if getattr(args, "camera_elevation", None) is not None else [0, 1, 2]
			for i in jitter_dims:
				cam.location[i] += _rand(args.camera_jitter)
		for lamp_name, jitter in (
			("Lamp_Key", args.key_light_jitter),
			("Lamp_Fill", args.fill_light_jitter),
			("Lamp_Back", args.back_light_jitter),
		):
			if jitter > 0:
				for i in range(3):
					bpy.data.objects[lamp_name].location[i] += _rand(jitter)

	def _compute_directions() -> Dict[str, Any]:
		# Similar to clevr-dataset-gen: use a temporary plane to define axes.
		try:
			bpy.ops.mesh.primitive_plane_add(size=5)
		except TypeError:
			# Older Blender versions
			bpy.ops.mesh.primitive_plane_add(radius=5)
		plane = bpy.context.object
		plane_normal = plane.data.vertices[0].normal
		camera = bpy.data.objects["Camera"]
		q = camera.matrix_world.to_quaternion()
		cam_behind = q @ Vector((0, 0, -1))
		cam_left = q @ Vector((-1, 0, 0))
		cam_up = q @ Vector((0, 1, 0))
		plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
		plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
		plane_up = cam_up.project(plane_normal).normalized()
		delete_object(plane)

		return {
			"behind": tuple(plane_behind),
			"front": tuple(-plane_behind),
			"left": tuple(plane_left),
			"right": tuple(-plane_left),
			"above": tuple(plane_up),
			"below": tuple(-plane_up),
		}

	def _place_objects(objects_spec: List[Dict[str, Any]], directions: Dict[str, Any]) -> List[Dict[str, Any]]:
		camera = bpy.data.objects["Camera"]
		blender_objects = []
		positions = []  # (x, y, r)
		objects_out = []

		for obj_i, spec in enumerate(objects_spec):
			shape = str(spec["shape"]).lower()
			color = str(spec["color"]).lower()
			size = str(spec.get("size", "small")).lower()
			material = str(spec.get("material", "rubber")).lower()

			if color not in color_name_to_rgba:
				raise ValueError(f"Unknown CLEVR color '{color}'")
			if shape not in shape_name_to_blend:
				raise ValueError(f"Unknown CLEVR shape '{shape}'")
			if material not in material_name_to_blend:
				raise ValueError(f"Unknown CLEVR material '{material}'")
			if size not in size_name_to_scale:
				raise ValueError(f"Unknown CLEVR size '{size}'")

			r = float(size_name_to_scale[size])
			obj_name = shape_name_to_blend[shape]
			rgba = color_name_to_rgba[color]
			mat_name = material_name_to_blend[material]

			# Find a placement that respects min_dist and margin.
			tries = 0
			while True:
				tries += 1
				if tries > args.max_retries:
					# Reset by deleting already-placed objects and restarting placement.
					for bo in blender_objects:
						delete_object(bo)
					return _place_objects(objects_spec, directions)

				x = random.uniform(-3, 3)
				y = random.uniform(-3, 3)

				# distance constraints
				dists_good = True
				margins_good = True
				for (xx, yy, rr) in positions:
					dx, dy = x - xx, y - yy
					dist = (dx * dx + dy * dy) ** 0.5
					if dist - r - rr < args.min_dist:
						dists_good = False
						break
					for direction_name in ["left", "right", "front", "behind"]:
						direction_vec = directions[direction_name]
						margin = dx * direction_vec[0] + dy * direction_vec[1]
						if 0 < margin < args.margin:
							margins_good = False
							break
					if not margins_good:
						break
				if dists_good and margins_good:
					break

			# rotation (radians)
			theta = 2.0 * math.pi * random.random()

			obj = add_object(shape_dir, obj_name, r, (x, y), theta=theta)
			blender_objects.append(obj)
			positions.append((x, y, r))

			# attach cached material
			try:
				mat = _get_or_create_material(mat_name, rgba)
				if hasattr(obj.data, "materials"):
					obj.data.materials.clear()
					obj.data.materials.append(mat)
			except Exception:
				# Fallback to upstream helper
				utils.add_material(mat_name, Color=rgba)

			pixel_coords = utils.get_camera_coords(camera, obj.location)
			objects_out.append(
				{
					"shape": shape,
					"size": size,
					"material": material,
					"3d_coords": tuple(obj.location),
					"rotation": theta,
					"pixel_coords": pixel_coords,
					"color": color,
				}
			)

		return objects_out

	with open(args.spec_jsonl, "r") as f:
		lines = [ln.strip() for ln in f if ln.strip()]

	for idx, ln in enumerate(lines):
		spec = json.loads(ln)
		seed = int(spec.get("seed", 0))
		random.seed(seed)

		image_filename = str(spec["image_filename"])
		image_path = os.path.join(args.output_image_dir, image_filename)

		# Cleanup previous iteration
		for obj in _created_objects:
			try:
				delete_object(obj)
			except Exception:
				pass
		_created_objects = []
		# Reset camera + lights to base pose, then jitter
		bpy.data.objects["Camera"].location = _cam0.copy()
		bpy.data.objects["Camera"].rotation_euler = _cam_rot0.copy()
		cam_elev = getattr(args, "camera_elevation", None)
		if cam_elev is not None:
			bpy.data.objects["Camera"].location[2] = float(cam_elev)
		cam_pitch_deg = getattr(args, "camera_pitch_deg", None)
		if cam_pitch_deg is not None:
			bpy.data.objects["Camera"].rotation_euler[0] = math.radians(float(cam_pitch_deg))
		bpy.data.objects["Lamp_Key"].location = _lk0.copy()
		bpy.data.objects["Lamp_Fill"].location = _lf0.copy()
		bpy.data.objects["Lamp_Back"].location = _lb0.copy()

		render_args.filepath = image_path

		_jitter_lights_and_camera()
		directions = _compute_directions()

		objects_spec = list(spec.get("objects", []))
		objects_out = _place_objects(objects_spec, directions)
		# best-effort: mark created objects for cleanup (they are the active objects added)
		try:
			# Heuristic: newest objects are at the end; grab the last N objects we placed.
			# This avoids traversing the whole scene.
			n_new = len(objects_spec)
			_created_objects = list(bpy.data.objects)[-n_new:]
		except Exception:
			_created_objects = []

		# Render
		bpy.ops.render.render(write_still=True)

		# Emit a parseable progress line for the parent process.
		print(f"__CLEVRGEN_PROGRESS__ {idx+1} {len(lines)}", flush=True)

		# Optional scene dump
		if args.output_scene_dir is not None:
			scene_out = {
				"split": str(spec.get("split", "train")),
				"image_index": int(spec.get("image_index", idx)),
				"image_filename": image_filename,
				"objects": objects_out,
				"directions": directions,
			}
			out_path = os.path.join(args.output_scene_dir, os.path.splitext(image_filename)[0] + ".json")
			with open(out_path, "w") as sf:
				json.dump(scene_out, sf)

		if (idx + 1) % 25 == 0:
			print(f"Rendered {idx+1}/{len(lines)}", flush=True)


if __name__ == "__main__":
	main()
