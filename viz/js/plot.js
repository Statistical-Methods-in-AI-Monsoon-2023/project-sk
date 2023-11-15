import * as THREE from 'three'
import { world } from './world.js'

function add_grid() {
	const { scene } = world

	const grid_opacity = 0.5

	const grid = new THREE.Group()
	var axes = new THREE.AxesHelper(1.2)
	grid.add(axes)

	var gridXZ = new THREE.GridHelper(1, 10)
	gridXZ.position.set(0.5, 0, 0.5)
	gridXZ.material.transparent = true
	gridXZ.material.opacity = grid_opacity
	grid.add(gridXZ)

	var gridXY = new THREE.GridHelper(1, 10)
	gridXY.position.set(0.5, 0.5, 0)
	gridXY.rotation.x = Math.PI / 2
	gridXY.material.transparent = true
	gridXY.material.opacity = grid_opacity
	grid.add(gridXY)

	var gridYZ = new THREE.GridHelper(1, 10)
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

function process_data(data, x = 1, y = -1, z = 1) {
	const geometry = new THREE.Geometry();
	const colors = [];

	// console.log(data)

	const width = data.length, height = data[0].length;
	let max_x = 0, max_y = 0, max_z = 0
	let min_x = Infinity, min_y = Infinity, min_z = Infinity

	data.forEach(function (col) {
		col.forEach(function (val) {
			max_x = Math.max(max_x, val.x)
			max_y = Math.max(max_y, val.y)
			max_z = Math.max(max_z, val.z)
			min_x = Math.min(min_x, val.x)
			min_y = Math.min(min_y, val.y)
			min_z = Math.min(min_z, val.z)
		})
	})
	// console.log(max_x, max_y, max_z, min_x, min_y, min_z)

	data.forEach(function (col) {
		col.forEach(function (val) {
			geometry.vertices.push(new THREE.Vector3(
				(val.x - min_x) / (max_x - min_x) * x,
				(val.y - min_y) / (max_y - min_y) * y,
				(val.z - min_z) / (max_z - min_z) * z,
			));
			colors.push(get_color((val.z - min_z) / (max_z - min_z) * z));
		});
	});

	const offset = (x, y) => x * width + y

	for (var x = 0; x < width - 1; x++) {
		for (var y = 0; y < height - 1; y++) {
			var vec0 = new THREE.Vector3(), vec1 = new THREE.Vector3(), n_vec = new THREE.Vector3();
			// one of two triangle polygons in one rectangle
			vec0.subVectors(geometry.vertices[offset(x, y)], geometry.vertices[offset(x + 1, y)]);
			vec1.subVectors(geometry.vertices[offset(x, y)], geometry.vertices[offset(x, y + 1)]);
			n_vec.crossVectors(vec0, vec1).normalize();
			geometry.faces.push(new THREE.Face3(offset(x, y), offset(x + 1, y), offset(x, y + 1), n_vec, [colors[offset(x, y)], colors[offset(x + 1, y)], colors[offset(x, y + 1)]]));
			// geometry.faces.push(new THREE.Face3(offset(x, y), offset(x, y + 1), offset(x + 1, y), n_vec.negate(), [colors[offset(x, y)], colors[offset(x, y + 1)], colors[offset(x + 1, y)]]));
			// the other one
			vec0.subVectors(geometry.vertices[offset(x + 1, y)], geometry.vertices[offset(x + 1, y + 1)]);
			vec1.subVectors(geometry.vertices[offset(x, y + 1)], geometry.vertices[offset(x + 1, y + 1)]);
			n_vec.crossVectors(vec0, vec1).normalize();
			geometry.faces.push(new THREE.Face3(offset(x + 1, y), offset(x + 1, y + 1), offset(x, y + 1), n_vec, [colors[offset(x + 1, y)], colors[offset(x + 1, y + 1)], colors[offset(x, y + 1)]]));
			// geometry.faces.push(new THREE.Face3(offset(x + 1, y), offset(x, y + 1), offset(x + 1, y + 1), n_vec.negate(), [colors[offset(x + 1, y)], colors[offset(x, y + 1)], colors[offset(x + 1, y + 1)]]));
		}
	}

	const material = new THREE.MeshStandardMaterial({
		vertexColors: THREE.VertexColors,
	});

	geometry.computeFaceNormals();
	geometry.computeVertexNormals();

	const mesh = new THREE.Mesh(geometry, material);
	// rotate the mesh to correct position
	mesh.rotation.x = -Math.PI / 2;

	// shadows
	mesh.castShadow = true
	mesh.receiveShadow = true

	// make wireframe
	const wireframe = new THREE.WireframeGeometry(geometry);
	const line = new THREE.LineSegments(wireframe);
	line.material.side = THREE.DoubleSide;
	line.material.opacity = 0.1;
	line.material.transparent = true;
	mesh.add(line);

	mesh.material.side = THREE.DoubleSide;
	mesh.material.opacity = 0.9;
	mesh.material.transparent = true;


	return {
		mesh,
		wireframe,
		extents: { min_x, min_y, min_z, max_x, max_y, max_z }
	}
}

function get_test_data() {
	var BEGIN = -10, END = 10;
	var data = new Array();
	for (var x = BEGIN; x < END; x++) {
		var row = [];
		for (var y = BEGIN; y < END; y++) {
			const z = 2.5 * (Math.cos(Math.sqrt(x * x + y * y)) + 1);
			row.push({ x: x, y: y, z: z });
		}
		data.push(row);
	}
	return data;
}

function load_model_name() {
	const { scene, plots,gui } = world
	// scene.add(get_mesh(get_test_data()))
	clear_plots()

	fetch(world.model_path_prefix + world.active_model).then(e => e.json()).then(array => {
		if(world.plotsFolder){
			world.plotsFolder.destroy()
		}
		world.plotsFolder= gui.addFolder("Plots")
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

function blend(geometry, data1, data2, alpha = 0.5) {
	const width = data1.length, height = data1[0].length
	// check if data1 and data2 have same shape
	if (data2.length != width || data2[0].length != height) {
		console.log("data1 and data2 have different shapes")
		return
	}
	const offset = (x, y) => x * width + y
	for (var x = 0; x < width - 1; x++) {
		for (var y = 0; y < height - 1; y++) {
			// update vertex position, blend data1 and data2
			geometry.vertices[offset(x, y)].z = data1[x][y] * alpha + data2[x][y] * (1 - alpha)
		}
	}
	geometry.verticesNeedUpdate = true
}

function add_minima_light(){
	const { scene } = world
	// minimaLight
	const minimaLight = new THREE.PointLight(0xffffff, 1, 100)
	// x, y at center of plot
	minimaLight.position.set(0.5, 0.1, 0.5)
	scene.add(minimaLight)

	// // add light helper
	// const minimaLightHelper = new THREE.PointLightHelper(minimaLight, 0.1);
	// scene.add(minimaLightHelper);

	world.minimaLight = minimaLight
}

function add_plot(){
	add_grid()
	add_minima_light()
}

export { load_model_name, update_plot, add_plot }