import * as THREE from 'three'
import { world } from './world.js'

function add_ground() {
	const { scene } = world

	const ground = new THREE.Mesh(
		new THREE.PlaneGeometry(500, 500),
		new THREE.MeshBasicMaterial({ color: 'green' }),
	)
	ground.rotation.x = -Math.PI / 2
	ground.position.y = -0.2
	scene.add(ground)
	ground.material.transparent = true
	ground.material.side = THREE.DoubleSide
	ground.material.opacity = 0.4

	var axes = new THREE.AxesHelper(1);
	scene.add(axes);

	var gridXZ = new THREE.GridHelper(1, 10);
	gridXZ.position.set(0.5, 0, 0.5);
	scene.add(gridXZ);

	var gridXY = new THREE.GridHelper(1, 10);
	gridXY.position.set(0.5, 0.5, 0);
	gridXY.rotation.x = Math.PI / 2;
	scene.add(gridXY);

	var gridYZ = new THREE.GridHelper(1, 10);
	gridYZ.position.set(0, 0.5, 0.5);
	gridYZ.rotation.z = Math.PI / 2;
	scene.add(gridYZ);
}

function get_color(val) {
	const from_color = new THREE.Color('brown')
	const to_color = new THREE.Color('red')
	return from_color.lerpHSL(to_color, val)
}

function get_mesh(data, x = 1, y = -1, z = 1) {
	var geometry = new THREE.Geometry();
	var colors = [];

	// console.log(data)

	var width = data.length, height = data[0].length;
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

	var offset = (x, y) => x * width + y

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

	var material = new THREE.MeshLambertMaterial({ vertexColors: THREE.VertexColors });
	var mesh = new THREE.Mesh(geometry, material);
	// rotate the mesh to correct position
	mesh.rotation.x = -Math.PI / 2;

	// make wireframe
	var wireframe = new THREE.WireframeGeometry(geometry);
	var line = new THREE.LineSegments(wireframe);
	line.material.opacity = 0.1;
	line.material.transparent = true;
	mesh.add(line);

	mesh.material.side = THREE.DoubleSide;
	mesh.material.opacity = 0.9;
	mesh.material.transparent = true;


	return mesh;
}

function get_test_data() {
	var BIGIN = -10, END = 10;
	var data = new Array();
	for (var x = BIGIN; x < END; x++) {
		var row = [];
		for (var y = BIGIN; y < END; y++) {
			const z = 2.5 * (Math.cos(Math.sqrt(x * x + y * y)) + 1);
			row.push({ x: x, y: y, z: z });
		}
		data.push(row);
	}
	return data;
}

function load_model_name() {
	const { scene, plots } = world
	// scene.add(get_mesh(get_test_data()))
	clear_plots()

	fetch(world.model_path_prefix + world.active_model).then(e => e.json()).then(array => {
		plots.loss.data = array.loss
		plots.acc.data = array.acc
		for (let plot in plots) {
			plots[plot].mesh = get_mesh(plots[plot].data)
		}
		update_plot()
	})
}

function clear_plots() {
	const { scene, plots } = world
	for (let plot in plots) {
		if (plots[plot].mesh) {
			scene.remove(plots[plot].mesh)
		}
	}
}

function update_plot() {
	const { scene, plots } = world
	clear_plots()
	for (let plot in plots) {
		if (plots[plot].visible) {
			scene.add(plots[plot].mesh)
		}
	}
}

function add_lights() {
	const { scene } = world
	const ambientLight = new THREE.AmbientLight(0xffffff, 5)
	scene.add(ambientLight)
}

function build_scene() {
	add_ground()
	load_model_name()
}

export { build_scene, add_lights, update_plot, load_model_name }