import * as THREE from 'three'
import { world } from './world.js'
import { get_pretty_name } from './gui.js'
import { destroy_physics, init_physics } from './physics.js'

function add_grid() {
	const { scene } = world

	const grid_opacity = 0.2
	const axes_opacity = 0.5

	const grid = new THREE.Group()
	const axes = new THREE.AxesHelper(1.2)
	axes.material.transparent = true
	axes.material.opacity = axes_opacity
	grid.add(axes)

	const gridXZ = new THREE.GridHelper(1, 10)
	gridXZ.position.set(0.5, 0, 0.5)
	gridXZ.material.transparent = true
	gridXZ.material.opacity = grid_opacity
	grid.add(gridXZ)

	const gridXY = new THREE.GridHelper(1, 10)
	gridXY.position.set(0.5, 0.5, 0)
	gridXY.rotation.x = Math.PI / 2
	gridXY.material.transparent = true
	gridXY.material.opacity = grid_opacity
	grid.add(gridXY)

	const gridYZ = new THREE.GridHelper(1, 10)
	gridYZ.position.set(0, 0.5, 0.5)
	gridYZ.rotation.z = Math.PI / 2
	gridYZ.material.transparent = true
	gridYZ.material.opacity = grid_opacity
	grid.add(gridYZ)

	grid.position.copy(world.origin)

	scene.add(grid)
	world.grid = grid
}

function get_color(val) {
	const from_color = new THREE.Color('brown')
	const to_color = new THREE.Color('red')
	return from_color.lerpHSL(to_color, val)
}

function process_data(data, log_plot = false) {
	const width = data.length, height = data[0].length
	const mesh = make_mesh(width, height)
	if (log_plot) {
		for (let x = 0; x < width; x++) {
			for (let y = 0; y < height; y++) {
				data[x][y].z = Math.log(1 + data[x][y].z)
			}
		}
	}
	normalize(data)
	add_noise(data)
	blend_mesh(mesh.geometry, data)
	// const cp = make_cp()
	// mesh.add(cp)
	// plot_contour(cp, data)
	return { mesh, width, height }
}

function add_noise(data) {
	const width = data.length, height = data[0].length
	// add noise to data
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			data[x][y].z += (Math.random() - 0.5) * 0.05
		}
	}
}

function normalize(data) {
	const width = data.length, height = data[0].length
	let max_x = 0, max_y = 0, max_z = 0
	let min_x = Infinity, min_y = Infinity, min_z = Infinity
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			max_x = Math.max(max_x, data[x][y].x)
			max_y = Math.max(max_y, data[x][y].y)
			max_z = Math.max(max_z, data[x][y].z)
			min_x = Math.min(min_x, data[x][y].x)
			min_y = Math.min(min_y, data[x][y].y)
			min_z = Math.min(min_z, data[x][y].z)
		}
	}

	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			data[x][y].x = (data[x][y].x - min_x) / (max_x - min_x)
			data[x][y].y = (data[x][y].y - min_y) / (max_y - min_y)
			data[x][y].z = (data[x][y].z - min_z) / (max_z - min_z)
		}
	}
	// console.log("max_x", max_x, "max_y", max_y, "max_z", max_z)
	// console.log("min_x", min_x, "min_y", min_y, "min_z", min_z)
}

function make_cp() {
	// add contour plane
	const contour_plane_geo = new THREE.PlaneGeometry(1, 1)
	const contour_plane_mat = new THREE.MeshStandardMaterial({
		// color: 0x000000,
		// transparent: true,
		// opacity: 0.2,
		side: THREE.DoubleSide,
	})
	const contour_plane = new THREE.Mesh(contour_plane_geo, contour_plane_mat)
	return contour_plane
}

function plot_contour(cp, data) {
	const texture = document.createElement('canvas')
	const ctx = texture.getContext('2d')
	const width = data.length, height = data[0].length
	// ctx.scale(width, height)
	for (let radius = 0; radius < width; radius++) {
		for (let angle = 0; angle < Math.PI * 2; angle += 0.01) {
			const x = Math.floor(radius * Math.cos(angle) + width / 2)
			const y = Math.floor(radius * Math.sin(angle) + height / 2)
			if (x < 0 || x >= width || y < 0 || y >= height) {
				continue
			}
			ctx.fillStyle = `rgba(255, 0, 0, ${data[x][y].z})`
			ctx.fillRect(x, y, 0.1, 0.1)
		}
	}
	ctx.fill()
	world.page_div.innerHTML = ""
	world.page_div.appendChild(texture)
	const textureLoader = new THREE.TextureLoader()
	const texture1 = textureLoader.load(texture.toDataURL())
	texture1.wrapS = THREE.RepeatWrapping
	texture1.wrapT = THREE.RepeatWrapping
	texture1.repeat.set(1, 1)
	cp.material.map = texture1
	cp.material.needsUpdate = true
}

function make_mesh(width = 20, height = 20) {
	const geometry = new THREE.Geometry()
	const colors = []

	for (let i = 0; i < width; i++) {
		for (let j = 0; j < height; j++) {
			geometry.vertices.push(new THREE.Vector3(i, j, 0))
			colors.push(get_color(0))
		}
	}

	const offset = (x, y) => x * width + y

	for (let x = 0; x < width - 1; x++) {
		for (let y = 0; y < height - 1; y++) {
			const vec0 = new THREE.Vector3(), vec1 = new THREE.Vector3(), n_vec = new THREE.Vector3()
			// one of two triangle polygons in one rectangle
			vec0.subVectors(geometry.vertices[offset(x, y)], geometry.vertices[offset(x + 1, y)])
			vec1.subVectors(geometry.vertices[offset(x, y)], geometry.vertices[offset(x, y + 1)])
			n_vec.crossVectors(vec0, vec1).normalize()
			geometry.faces.push(new THREE.Face3(offset(x, y), offset(x + 1, y), offset(x, y + 1), n_vec, [colors[offset(x, y)], colors[offset(x + 1, y)], colors[offset(x, y + 1)]]))
			// geometry.faces.push(new THREE.Face3(offset(x, y), offset(x, y + 1), offset(x + 1, y), n_vec.negate(), [colors[offset(x, y)], colors[offset(x, y + 1)], colors[offset(x + 1, y)]]))
			// the other one
			vec0.subVectors(geometry.vertices[offset(x + 1, y)], geometry.vertices[offset(x + 1, y + 1)])
			vec1.subVectors(geometry.vertices[offset(x, y + 1)], geometry.vertices[offset(x + 1, y + 1)])
			n_vec.crossVectors(vec0, vec1).normalize()
			geometry.faces.push(new THREE.Face3(offset(x + 1, y), offset(x + 1, y + 1), offset(x, y + 1), n_vec, [colors[offset(x + 1, y)], colors[offset(x + 1, y + 1)], colors[offset(x, y + 1)]]))
			// geometry.faces.push(new THREE.Face3(offset(x + 1, y), offset(x, y + 1), offset(x + 1, y + 1), n_vec.negate(), [colors[offset(x + 1, y)], colors[offset(x, y + 1)], colors[offset(x + 1, y + 1)]]))
		}
	}

	const material = new THREE.MeshStandardMaterial({
		vertexColors: THREE.VertexColors,
		roughness: 0.5
	})

	geometry.computeFaceNormals()
	geometry.computeVertexNormals()

	const mesh = new THREE.Mesh(geometry, material)
	// rotate the mesh
	mesh.rotation.x = -Math.PI / 2
	mesh.rotation.z = -Math.PI / 2

	mesh.position.copy(world.origin)

	// shadows
	mesh.castShadow = true
	mesh.receiveShadow = true

	// make wireframe
	const line = new THREE.LineSegments(geometry)
	line.material.side = THREE.DoubleSide
	line.material.opacity = 0.1
	line.material.transparent = true
	mesh.add(line)

	mesh.material.side = THREE.DoubleSide
	mesh.material.opacity = 0.9
	mesh.material.transparent = true

	return mesh
}

function get_test_data() {
	const BEGIN = -10, END = 10
	const data = new Array()
	for (let x = BEGIN; x < END; x++) {
		const row = []
		for (let y = BEGIN; y < END; y++) {
			const z = 2.5 * (Math.cos(Math.sqrt(x * x + y * y)) + 1)
			row.push({ x: x, y: y, z: z })
		}
		data.push(row)
	}
	return data
}

function load_model_name() {
	const { scene, plots, gui, blend_plot } = world
	// world.plot.(get_mesh(get_test_data()))

	fetch(world.model_path_prefix + world.active_model).then(e => e.json()).then(array => {
		clear_plots()
		if (world.plotsFolder) {
			world.plotsFolder.destroy()
		}
		world.plotsFolder = gui.addFolder("Plots")
		world.plotsFolder.add(world, 'multiple_plots').name("Multiple Plots").onChange(() => update_plot())
		let first = true
		for (let plot_name in array) {
			plots[plot_name] = {
				data: array[plot_name],
				processed: process_data(array[plot_name], world.log_plot),
				visible: first,
			}
			if (first) {
				if (!blend_plot.processed) {
					blend_plot.data = array[plot_name]
					blend_plot.processed = process_data(array[plot_name])
					// blend_plot.processed.mesh.material.color = new THREE.Color('darkred')
					world.plot.add(blend_plot.processed.mesh)
				}
				else if (plots[plot_name].processed.width != blend_plot.processed.width
					|| plots[plot_name].processed.height != blend_plot.processed.height
				) {
					console.log("data have different shapes")
					blend_plot.data = array[plot_name]
					scene.remove(blend_plot.processed.mesh)
					blend_plot.processed = process_data(array[plot_name])
					// blend_plot.processed.mesh.material.color = new THREE.Color('darkred')
					world.plot.add(blend_plot.processed.mesh)
				} else {
					console.log("data have same shapes")
				}
				first = false
				world.active_plot = plot_name
			}
			// add to folder
			world.plotsFolder.add(plots[plot_name], 'visible')
				.name(get_pretty_name(plot_name))
				.onChange(() => update_plot(plot_name))
				.listen()
		}
		update_plot(world.active_plot)
	})
}

function clear_plots() {
	const { scene, plots } = world
	for (let plot in plots) {
		if (plots[plot].processed?.mesh) {
			scene.remove(plots[plot].processed.mesh)
		}
	}
	if (world.blend_plot.processed) {
		world.blend_plot.processed.mesh.visible = false
	}
}

function update_plot(plot_name) {
	const { scene, plots, blend_plot } = world
	clear_plots()
	blend_plot.processed.mesh.visible = !world.multiple_plots
	if (world.multiple_plots) {
		let count = 0
		for (let plot in plots) {
			if (plots[plot].visible) {
				world.plot.add(plots[plot].processed.mesh)
				world.active_plot = plot
				count++
			}
		}
		if (count == 1) {
			blend_plot.data = plots[world.active_plot].data
		}
		return
	}
	for (let plot in plots) {
		if (plots[plot].visible) {
			plot_name ||= plot
			plots[plot].visible = false
		}
	}
	plot_name ||= Object.keys(plots)[0]
	if (!plot_name) {
		return
	}

	plots[plot_name].visible = true
	world.active_plot = plots[plot_name]
	destroy_physics()
	swap_plot(plot_name)
}

function swap_plot(plot_name) {
	const { plots, blend_plot } = world
	const plot = plots[plot_name]
	if (plot.processed.width != blend_plot.processed.width
		|| plot.processed.height != blend_plot.processed.height) {
		console.log("data have different shapes")
		return
	}
	blend_mesh_animate(plot.data)
}

function blend_mesh_animate(data) {
	const { blend_plot } = world
	let alpha = 0
	blend_plot.processed.mesh.visible = true
	// check if data1 and data2 have same shape
	if (blend_plot.data.length != data.length || blend_plot.data[0].length != data[0].length) {
		console.log("data1 and data2 have different shapes")
		return
	}
	cancelAnimationFrame(world.blend_plot.frameId)
	const old_data = blend_plot.data
	function animate() {
		if (alpha < 1) {
			alpha += 0.01
			blend_mesh(blend_plot.processed.mesh.geometry, data, old_data, alpha)
			world.blend_plot.frameId = requestAnimationFrame(animate)
		} else {
			blend_plot.data = data
			init_physics()
		}
	}
	animate()
}

function blend_mesh(geometry, data1, data2, alpha = 0.5) {
	const width = data1.length, height = data1[0].length
	if (!data2) {
		data2 = data1
	}
	// check if data1 and data2 have same shape
	if (data2.length != width || data2[0].length != height) {
		console.log("data1 and data2 have different shapes")
		return
	}
	const offset = (x, y) => x * width + y
	for (let x = 0; x < width; x++) {
		for (let y = 0; y < height; y++) {
			// update vertex position, blend data1 and data2
			geometry.vertices[offset(x, y)].x = data1[x][y].x * alpha + data2[x][y].x * (1 - alpha)
			geometry.vertices[offset(x, y)].y = data1[x][y].y * alpha + data2[x][y].y * (1 - alpha)
			geometry.vertices[offset(x, y)].z = data1[x][y].z * alpha + data2[x][y].z * (1 - alpha)
		}
	}

	// update vertex colors
	const colors = []
	for (let i = 0; i < geometry.vertices.length; i++) {
		colors.push(get_color(geometry.vertices[i].z))
	}
	geometry.faces.forEach(face => {
		face.vertexColors = [colors[face.a], colors[face.b], colors[face.c]]
	})
	geometry.colorsNeedUpdate = true
	geometry.verticesNeedUpdate = true
	geometry.computeFaceNormals()
	geometry.computeVertexNormals()
}

function add_minima_light() {
	const { scene } = world
	// minimaLight
	const minimaLight = new THREE.PointLight(0xffffff, 1, 100)
	// x, y at center of plot
	minimaLight.position.copy(world.origin)
	minimaLight.position.x += 0.5
	minimaLight.position.z += 0.5
	minimaLight.position.y += 0.1
	scene.add(minimaLight)

	// // add light helper
	// const minimaLightHelper = new THREE.PointLightHelper(minimaLight, 0.1)
	// scene.add(minimaLightHelper)

	world.minimaLight = minimaLight
}

function add_plot() {
	const { scene } = world
	const plot = new THREE.Group()
	scene.add(plot)
	world.plot = plot

	add_grid()
	add_minima_light()
}

export { load_model_name, update_plot, add_plot }