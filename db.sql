  /*
  SQLyog Enterprise - MySQL GUI v6.56
  MySQL - 5.5.5-10.1.13-MariaDB : Database - pavement
  *********************************************************************
  */


  /*!40101 SET NAMES utf8 */;

  /*!40101 SET SQL_MODE=''*/;

  /*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
  /*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

  CREATE DATABASE /*!32312 IF NOT EXISTS*/`pavement` /*!40100 DEFAULT CHARACTER SET latin1 */;

  USE `pavement`;

  /*Table structure for table `ureg` */

  DROP TABLE IF EXISTS `ureg`;

  CREATE TABLE `ureg` (
    `id` int(10) NOT NULL AUTO_INCREMENT,
    `name` varchar(100) DEFAULT NULL,
    `email` varchar(100) DEFAULT NULL,
    `pwd` varchar(100) DEFAULT NULL,
    `cpwd` varchar(100) DEFAULT NULL,
    `pno` varchar(100) DEFAULT NULL,
    PRIMARY KEY (`id`)
  ) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

  /*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
  /*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
