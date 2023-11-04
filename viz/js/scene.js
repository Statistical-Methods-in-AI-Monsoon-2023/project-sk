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
}

function get_color(max, min, val) {
	var MIN_L = 40, MAX_L = 100;
	var color = new THREE.Color();
	var h = 0 / 240;
	var s = 80 / 240;
	var l = (((MAX_L - MIN_L) / (max - min)) * val) / 240;
	color.setHSL(h, s, l);
	return color;
}

function get_mesh(data) {
	var geometry = new THREE.Geometry();
	var colors = [];

	console.log(data)

	var width = data.length, height = data[0].length;
	data.forEach(function (col) {
		col.forEach(function (val) {
			geometry.vertices.push(new THREE.Vector3(val.x, val.y, val.z))
			colors.push(get_color(2.5, 0, val.z));
		});
	});

	var offset = function (x, y) {
		return x * width + y;
	}

	for (var x = 0; x < width - 1; x++) {
		for (var y = 0; y < height - 1; y++) {
			var vec0 = new THREE.Vector3(), vec1 = new THREE.Vector3(), n_vec = new THREE.Vector3();
			// one of two triangle polygons in one rectangle
			vec0.subVectors(geometry.vertices[offset(x, y)], geometry.vertices[offset(x + 1, y)]);
			vec1.subVectors(geometry.vertices[offset(x, y)], geometry.vertices[offset(x, y + 1)]);
			n_vec.crossVectors(vec0, vec1).normalize();
			geometry.faces.push(new THREE.Face3(offset(x, y), offset(x + 1, y), offset(x, y + 1), n_vec, [colors[offset(x, y)], colors[offset(x + 1, y)], colors[offset(x, y + 1)]]));
			geometry.faces.push(new THREE.Face3(offset(x, y), offset(x, y + 1), offset(x + 1, y), n_vec.negate(), [colors[offset(x, y)], colors[offset(x, y + 1)], colors[offset(x + 1, y)]]));
			// the other one
			vec0.subVectors(geometry.vertices[offset(x + 1, y)], geometry.vertices[offset(x + 1, y + 1)]);
			vec1.subVectors(geometry.vertices[offset(x, y + 1)], geometry.vertices[offset(x + 1, y + 1)]);
			n_vec.crossVectors(vec0, vec1).normalize();
			geometry.faces.push(new THREE.Face3(offset(x + 1, y), offset(x + 1, y + 1), offset(x, y + 1), n_vec, [colors[offset(x + 1, y)], colors[offset(x + 1, y + 1)], colors[offset(x, y + 1)]]));
			geometry.faces.push(new THREE.Face3(offset(x + 1, y), offset(x, y + 1), offset(x + 1, y + 1), n_vec.negate(), [colors[offset(x + 1, y)], colors[offset(x, y + 1)], colors[offset(x + 1, y + 1)]]));
		}
	}

	var material = new THREE.MeshLambertMaterial({ vertexColors: THREE.VertexColors });
	var mesh = new THREE.Mesh(geometry, material);
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

function add_plot() {
	const { scene } = world

	scene.add(get_mesh(get_test_data()))

	fetch("loss.json").then(e => e.json()).then(array => {
		scene.add(get_mesh(array))
	})

}


function add_lights() {
	const { scene } = world
	const ambientLight = new THREE.AmbientLight(0xffffff, 5)
	scene.add(ambientLight)
}

function build_scene() {
	add_ground()
	add_plot()
}

export { build_scene, add_lights }