import * as THREE from 'three'
import { world } from './world.js'

function add_grid() {
	const { scene } = world

	const grid_opacity = 0.5

	const grid = new THREE.Group()
	const axes = new THREE.AxesHelper(1.2)
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

	scene.add(grid)
}


function get_color(val) {
	const from_color = new THREE.Color('brown')
	const to_color = new THREE.Color('red')
	return from_color.lerpHSL(to_color, val)
}

function process_data1(data, x = 1, y = -1, z = 1) {
	const surface = make_surface()
	blend_surface(surface.mesh.geometry, data)
	return surface
}

function normalize_inplace(data) {
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
}

function process_data(data, x = 1, y = -1, z = 1) {
	const geometry = new THREE.Geometry()
	const colors = []

	const width = data.length, height = data[0].length
	normalize_inplace(data)

	data.forEach(function (col) {
		col.forEach(function (val) {
			geometry.vertices.push(new THREE.Vector3(val.x * x, val.y * y, val.z * z))
			colors.push(get_color(val.z * z))
		})
	})

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
	})

	geometry.computeFaceNormals()
	geometry.computeVertexNormals()

	const mesh = new THREE.Mesh(geometry, material)
	// rotate the mesh to correct position
	mesh.rotation.x = -Math.PI / 2

	// shadows
	mesh.castShadow = true
	mesh.receiveShadow = true

	// make wireframe
	const wireframe = new THREE.WireframeGeometry(geometry)
	const line = new THREE.LineSegments(wireframe)
	line.material.side = THREE.DoubleSide
	line.material.opacity = 0.1
	line.material.transparent = true
	mesh.add(line)

	mesh.material.side = THREE.DoubleSide
	mesh.material.opacity = 0.9
	mesh.material.transparent = true


	return {
		mesh,
		wireframe,
	}
}

function make_surface(width = 20, height = 20) {
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
	})

	geometry.computeFaceNormals()
	geometry.computeVertexNormals()

	const mesh = new THREE.Mesh(geometry, material)
	// rotate the mesh to correct position
	mesh.rotation.x = -Math.PI / 2

	// shadows
	mesh.castShadow = true
	mesh.receiveShadow = true

	// make wireframe
	const wireframe = new THREE.WireframeGeometry(geometry)
	const line = new THREE.LineSegments(wireframe)
	line.material.side = THREE.DoubleSide
	line.material.opacity = 0.1
	line.material.transparent = true
	mesh.add(line)

	mesh.material.side = THREE.DoubleSide
	mesh.material.opacity = 0.9
	mesh.material.transparent = true


	return {
		mesh,
		wireframe,
	}
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
	const { scene, plots, gui } = world
	// scene.add(get_mesh(get_test_data()))
	clear_plots()

	fetch(world.model_path_prefix + world.active_model).then(e => e.json()).then(array => {
		if (world.plotsFolder) {
			world.plotsFolder.destroy()
		}
		world.plotsFolder = gui.addFolder("Plots")
		let first = true
		for (let data in array) {
			plots[data] = {
				data: array[data],
				processed: process_data(array[data]),
				visible: first,
			}
			first = false
			// add to folder
			world.plotsFolder.add(plots[data], 'visible').name(data).onChange(update_plot)
		}
		update_plot()
	})
}

function clear_plots() {
	const { scene, plots } = world
	for (let plot in plots) {
		if (plots[plot].processed?.mesh) {
			scene.remove(plots[plot].processed.mesh)
		}
	}
}

function update_plot() {
	const { scene, plots } = world
	clear_plots()
	for (let plot in plots) {
		if (plots[plot].visible) {
			scene.add(plots[plot].processed.mesh)
		}
	}
}

function blend_surface(geometry, data1, data2, alpha = 0.5) {
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
	for (let x = 0; x < width - 1; x++) {
		for (let y = 0; y < height - 1; y++) {
			// update vertex position, blend data1 and data2
			geometry.vertices[offset(x, y)].z = data1[x][y] * alpha + data2[x][y] * (1 - alpha)
		}
	}
	geometry.verticesNeedUpdate = true
	geometry.computeFaceNormals()
	geometry.computeVertexNormals()
}

function add_minima_light() {
	const { scene } = world
	// minimaLight
	const minimaLight = new THREE.PointLight(0xffffff, 1, 100)
	// x, y at center of plot
	minimaLight.position.set(0.5, 0.1, 0.5)
	scene.add(minimaLight)

	// // add light helper
	// const minimaLightHelper = new THREE.PointLightHelper(minimaLight, 0.1)
	// scene.add(minimaLightHelper)

	world.minimaLight = minimaLight
}

function add_plot() {
	add_grid()
	add_minima_light()
}

export { load_model_name, update_plot, add_plot }